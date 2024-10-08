//! A libfuzzer-like fuzzer with llmp-multithreading support and restarts
//! The example harness is built for libpng.
//! In this example, you will see the use of the `launcher` feature.
//! The `launcher` will spawn new processes for each cpu core.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use libafl::observers::CanTrack;
use libafl_bolts::{
    current_nanos,
    os::dup2,
    rands::StdRand,
    shmem::{ShMemProvider, StdShMemProvider},
    tuples::tuple_list,
};

use clap::{Arg, Command};
use core::time::Duration;
#[cfg(unix)]
use nix::{self, unistd::dup};
#[cfg(unix)]
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::{
    env,
    fs::{self, File},
    io::{self, Read, Write},
    path::PathBuf,
};

use libafl::{
    corpus::{Corpus, InMemoryOnDiskCorpus, OnDiskCorpus},
    events::SimpleRestartingEventManager,
    executors::{inprocess::InProcessExecutor, ExitKind},
    feedback_or,
    feedbacks::{CrashFeedback, MaxMapFeedback, TimeFeedback},
    fuzzer::{Fuzzer, StdFuzzer},
    generators::{Generator, NautilusContext, NautilusGenerator},
    inputs::{
        EncodedInput, Input, InputDecoder, InputEncoder, NaiveTokenizer, TokenInputEncoderDecoder,
    },
    monitors::SimpleMonitor,
    mutators::{encoded_mutations::encoded_mutations, StdScheduledMutator},
    observers::{HitcountsMapObserver, TimeObserver},
    schedulers::{IndexesLenTimeMinimizerScheduler, QueueScheduler},
    stages::mutational::StdMutationalStage,
    state::{HasCorpus, StdState},
    Error, Evaluator,
};

use libafl_targets::{libfuzzer_initialize, libfuzzer_test_one_input, std_edges_map_observer};

/// The fuzzer main (as `no_mangle` C function)
#[no_mangle]
pub fn libafl_main() {
    if env::args().len() <= 1 {
        return;
    }

    // Registry the metadata types used in this fuzzer
    // Needed only on no_std
    //RegistryBuilder::register::<Tokens>();

    let res = match Command::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author("AFLplusplus team")
        .about("LibAFL-based fuzzer for Fuzzbench")
        .arg(
            Arg::new("out")
                .short('o')
                .long("output")
                .help("The directory to place finds in ('corpus')"),
        )
        .arg(
            Arg::new("report")
                .short('r')
                .long("report")
                .help("The directory to place dumped testcases ('corpus')"),
        )
        .arg(
            Arg::new("grammar")
                .short('g')
                .long("grammar")
                .help("The grammar model"),
        )
        .arg(
            Arg::new("timeout")
                .short('t')
                .long("timeout")
                .help("Timeout for each individual execution, in milliseconds")
                .default_value("12000"),
        )
        .arg(Arg::new("remaining"))
        .try_get_matches()
    {
        Ok(res) => res,
        Err(err) => {
            println!(
                "Syntax: {}, -o corpus_dir -g grammar.json\n{:?}",
                env::current_exe()
                    .unwrap_or_else(|_| "fuzzer".into())
                    .to_string_lossy(),
                err,
            );
            return;
        }
    };

    println!(
        "Workdir: {:?}",
        env::current_dir().unwrap().to_string_lossy().to_string()
    );

    // For fuzzbench, crashes and finds are inside the same `corpus` directory, in the "queue" and "crashes" subdir.
    let mut out_dir = PathBuf::from(
        res.get_one::<String>("out")
            .expect("The --output parameter is missing")
            .to_string(),
    );
    if fs::create_dir(&out_dir).is_err() {
        println!("Out dir at {:?} already exists.", &out_dir);
        if !out_dir.is_dir() {
            println!("Out dir at {:?} is not a valid directory!", &out_dir);
            return;
        }
    }
    let mut initial_dir = out_dir.clone();
    initial_dir.push("initial");

    if let Some(filenames) = res.get_many::<String>("remaining") {
        let filenames: Vec<&str> = filenames.map(String::as_str).collect();
        if !filenames.is_empty() {
            run_testcases(initial_dir, &filenames);
            return;
        }
    }

    let report_dir = PathBuf::from(
        res.get_one::<String>("report")
            .expect("The --report parameter is missing")
            .to_string(),
    );
    if fs::create_dir(&report_dir).is_err() {
        println!("Report dir at {:?} already exists.", &report_dir);
        if !report_dir.is_dir() {
            println!("Report dir at {:?} is not a valid directory!", &report_dir);
            return;
        }
    }

    let grammar_path = PathBuf::from(
        res.get_one::<String>("grammar")
            .expect("The --grammar parameter is missing")
            .to_string(),
    );
    if !grammar_path.is_file() {
        println!("{:?} is not a valid file!", &grammar_path);
        return;
    }
    let context = NautilusContext::from_file(64, grammar_path);

    fs::create_dir_all(&initial_dir).unwrap();
    let mut crashes = out_dir.clone();
    crashes.push("crashes");
    out_dir.push("queue");

    let timeout = Duration::from_millis(
        res.get_one::<String>("timeout")
            .unwrap()
            .to_string()
            .parse()
            .expect("Could not parse timeout in milliseconds"),
    );

    fuzz(initial_dir, out_dir, crashes, report_dir, context, timeout)
        .expect("An error occurred while fuzzing");
}

fn run_testcases(initial_dir: PathBuf, filenames: &[&str]) {
    // The actual target run starts here.
    // Call LLVMFUzzerInitialize() if present.
    let args: Vec<String> = env::args().collect();
    if unsafe { libfuzzer_initialize(&args) } == -1 {
        println!("Warning: LLVMFuzzerInitialize failed with -1")
    }

    let mut tokenizer = NaiveTokenizer::default();
    let mut encoder_decoder = TokenInputEncoderDecoder::new();
    for i in 0..4096 {
        let mut file =
            fs::File::open(&initial_dir.join(format!("id_{}", i))).expect("no file found");
        let mut buffer = vec![];
        file.read_to_end(&mut buffer).expect("buffer overflow");

        let _ = encoder_decoder
            .encode(&buffer, &mut tokenizer)
            .expect("encoding failed");
    }

    println!(
        "You are not fuzzing, just executing {} testcases",
        filenames.len()
    );
    for fname in filenames {
        println!("Executing {}", fname);

        let input = EncodedInput::from_file(fname).unwrap();

        let mut bytes = vec![];
        encoder_decoder.decode(&input, &mut bytes).unwrap();
        if *bytes.last().unwrap() != 0 {
            bytes.push(0);
        }
        unsafe {
            println!("Testcase: {}", std::str::from_utf8_unchecked(&bytes));
        }
        unsafe { libfuzzer_test_one_input(&bytes) };
    }
}

/// The actual fuzzer
fn fuzz(
    initial_dir: PathBuf,
    corpus_dir: PathBuf,
    objective_dir: PathBuf,
    report_dir: PathBuf,
    context: NautilusContext,
    timeout: Duration,
) -> Result<(), Error> {
    #[cfg(unix)]
    let mut stdout_cpy = unsafe {
        let new_fd = dup(io::stdout().as_raw_fd())?;
        File::from_raw_fd(new_fd)
    };
    #[cfg(unix)]
    let file_null = File::open("/dev/null")?;

    // 'While the monitor are state, they are usually used in the broker - which is likely never restarted
    let monitor = SimpleMonitor::new(|s| {
        #[cfg(unix)]
        writeln!(&mut stdout_cpy, "{}", s).unwrap();
        #[cfg(windows)]
        println!("{}", s);
    });

    let mut tokenizer = NaiveTokenizer::default();
    let mut encoder_decoder = TokenInputEncoderDecoder::new();

    let mut generator = NautilusGenerator::new(&context);

    // We need a shared map to store our state before a crash.
    // This way, we are able to continue fuzzing afterwards.
    let mut shmem_provider = StdShMemProvider::new()?;

    let (state, mut mgr) = match SimpleRestartingEventManager::launch(monitor, &mut shmem_provider)
    {
        // The restarting state will spawn the same process again as child, then restarted it each time it crashes.
        Ok(res) => res,
        Err(err) => match err {
            Error::ShuttingDown => {
                return Ok(());
            }
            _ => {
                panic!("Failed to setup the restarter: {}", err);
            }
        },
    };

    // Create an observation channel using the coverage map
    let edges_observer =
        HitcountsMapObserver::new(unsafe { std_edges_map_observer("edges") }).track_indices();

    // Create an observation channel to keep track of the execution time
    let time_observer = TimeObserver::new("time");

    // Feedback to rate the interestingness of an input
    // This one is composed by two Feedbacks in OR
    let mut feedback = feedback_or!(
        // New maximization map feedback linked to the edges observer and the feedback state
        MaxMapFeedback::new(&edges_observer),
        // Time feedback, this one does not need a feedback state
        TimeFeedback::new(&time_observer)
    );

    // A feedback to choose if an input is a solution or not
    let mut objective = CrashFeedback::new();

    // If not restarting, create a State from scratch
    let mut state = state.unwrap_or_else(|| {
        StdState::new(
            // RNG
            StdRand::with_seed(current_nanos()),
            // Corpus that will be evolved, we keep it in memory for performance
            InMemoryOnDiskCorpus::new(corpus_dir).unwrap(),
            // Corpus in which we store solutions (crashes in this example),
            // on disk so the user can get them after stopping the fuzzer
            OnDiskCorpus::new(objective_dir).unwrap(),
            // States of the feedbacks.
            // They are the data related to the feedbacks that you want to persist in the State.
            &mut feedback,
            &mut objective,
        )
        .unwrap()
    });

    let mut bytes = vec![];
    let mut initial_inputs = vec![];
    for i in 0..4096 {
        let nautilus = generator.generate(&mut state).unwrap();
        nautilus.unparse(&context, &mut bytes);

        let mut file = fs::File::create(&initial_dir.join(format!("id_{}", i))).unwrap();
        file.write_all(&bytes).unwrap();

        let input = encoder_decoder
            .encode(&bytes, &mut tokenizer)
            .expect("encoding failed");
        initial_inputs.push(input);
    }

    // A minimization+queue policy to get testcasess from the corpus
    let scheduler = IndexesLenTimeMinimizerScheduler::new(&edges_observer, QueueScheduler::new());

    // A fuzzer with feedbacks and a corpus scheduler
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    // The wrapped harness function, calling out to the LLVM-style harness
    let mut harness = |input: &EncodedInput| {
        bytes.clear();
        encoder_decoder.decode(input, &mut bytes).unwrap();
        unsafe { libfuzzer_test_one_input(&bytes) };
        ExitKind::Ok
    };

    // Create the executor for an in-process function with one observer for edge coverage and one for the execution time
    let mut executor = InProcessExecutor::with_timeout(
        &mut harness,
        tuple_list!(edges_observer, time_observer),
        &mut fuzzer,
        &mut state,
        &mut mgr,
        timeout,
    )?;

    // The actual target run starts here.
    // Call LLVMFUzzerInitialize() if present.
    let args: Vec<String> = env::args().collect();
    if unsafe { libfuzzer_initialize(&args) } == -1 {
        println!("Warning: LLVMFuzzerInitialize failed with -1")
    }

    // In case the corpus is empty (on first run), reset
    if state.corpus().count() < 1 {
        for input in initial_inputs {
            fuzzer
                .add_input(&mut state, &mut executor, &mut mgr, input)
                .unwrap();
        }
    }

    // Setup a basic mutator with a mutational stage
    let mutator = StdScheduledMutator::new(encoded_mutations());

    let fuzzbench = libafl::stages::DumpToDiskStage::new(
        |input: &EncodedInput, _stage: &_| {
            let mut bytes = vec![];
            encoder_decoder.decode(input, &mut bytes).unwrap();
            bytes
        },
        &report_dir.join("queue"),
        &report_dir.join("crashes"),
    )
    .unwrap();

    let mut stages = tuple_list!(fuzzbench, StdMutationalStage::new(mutator));

    // Remove target ouput (logs still survive)
    #[cfg(unix)]
    {
        let null_fd = file_null.as_raw_fd();
        dup2(null_fd, io::stdout().as_raw_fd())?;
        dup2(null_fd, io::stderr().as_raw_fd())?;
    }

    fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
    Ok(())
}

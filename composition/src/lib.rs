// There are just two options for mutators
// grimoire or mopt. I'm too lazy to put these two into the same file!

#[cfg(feature = "grimoire")]
pub mod grimoire;

#[cfg(feature = "mopt")]
pub mod mopt;
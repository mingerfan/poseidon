# Metadata only; never fetch packages or import the floating default channel.
{
  nixVersion = builtins.nixVersion;
  system = builtins.currentSystem;
  home = builtins.getEnv "HOME";
  portableLocation = builtins.getEnv "NP_LOCATION";
  runtime = builtins.getEnv "NP_RUNTIME";
  git = builtins.getEnv "NP_GIT";
  storeVisibleInside = builtins.pathExists "/nix/store";
  cacheHome = builtins.getEnv "XDG_CACHE_HOME";
  tempDirectory = builtins.getEnv "TMPDIR";
}

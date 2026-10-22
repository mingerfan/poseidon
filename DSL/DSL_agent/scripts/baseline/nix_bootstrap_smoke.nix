# Self-contained bootstrap smoke. No nixpkgs import, fetcher or external inputs.
builtins.derivation {
  name = "poseidon-nix-bootstrap-smoke";
  system = builtins.currentSystem;
  builder = "/bin/sh";
  args = [ "-c" ''printf '%s\n' 'poseidon-nix-bootstrap-ok' > "$out"'' ];
}

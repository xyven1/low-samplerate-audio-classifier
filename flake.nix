{
  description = "Low samplerate audio classifier";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs = {nixpkgs, ...}: let
    allSystems = [
      "aarch64-darwin"
      "aarch64-linux"
      "x86_64-linux"
    ];
    forEachSystem = f:
      nixpkgs.lib.genAttrs allSystems (system:
        f {
          inherit system;
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };
        });
  in {
    devShells = forEachSystem ({
      pkgs,
      system,
    }: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          (python3.withPackages (p:
            with p; [
              debugpy
              numpy
              matplotlib
              torch-bin
              librosa
            ]))
          (writeShellScriptBin "debugpy-adapter" ''
            exec "python" -m debugpy.adapter "$@"
          '')
          basedpyright
          ruff
        ];
      };
    });
  };
}

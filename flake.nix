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
      default = let
        # libtorch-bin = pkgs.libtorch-bin.overrideAttrs (old: {
        #   version = "2.6.0";
        #   src = pkgs.fetchzip {
        #     url = "https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2bcu126.zip";
        #     sha256 = "";
        #   };
        #   cudaSupport = true;
        # });
        # combined = pkgs.symlinkJoin {
        #   name = "libtorch";
        #   paths = [
        #     libtorch-bin
        #     libtorch-bin.dev
        #   ];
        # };
      in
        pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            clippy
            rust-analyzer
            rustfmt
            (writeShellScriptBin "lldb-dap" ''
              ${pkgs.lib.getExe' pkgs.lldb "lldb-dap"} --pre-init-command  "command script import ${pkgs.fetchFromGitHub {
                owner = "cmrschwarz";
                repo = "rust-prettifier-for-lldb";
                rev = "v0.4";
                hash = "sha256-eje+Bs7kS87x9zCwH+7Tl1S/Bdv8dGkA0BoijOOdmeI=";
              }}/rust_prettifier_for_lldb.py" $@
            '')
          ];
          LD_LIBRARY_PATH = "${
            pkgs.symlinkJoin {
              name = "vulkan-deps";
              paths = with pkgs; [
                vulkan-loader
              ];
            }
          }/lib";
        };
    });
  };
}

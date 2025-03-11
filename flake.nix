# flake.nix
{
  description = "An (impure) flake for Python development.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

    python = pkgs.python313Full;

    fhs = pkgs.buildFHSEnv {
      name = "python-fhs-env";
      targetPkgs = pkgs: with pkgs; (
        [
          (python.withPackages(
            pp: with pp; [ pip ]
          ))
        ]
      );
      runScript = "bash";
      profile = ''
        export PYTHONPATH="${builtins.toString ./.}"
        export MPLBACKEND="TkAgg"
        export DISPLAY=":0.0"
        export PS1="\n\[\033[1;32m\][\[\e]0;\u@\h:\w\a\]\u@\033[1;33mFHS\033[1;32m:\w]\$\[\033[0m\]"
        python -m venv .venv
        source .venv/bin/activate
        pip install pip poetry-core setuptools setuptools-scm
        pip install jax[cuda]
        pip install .[dev,docs]
        echo "Welcome to the $(python -V) FHS/venv shell, with interpreter $(which python)"
      '';
    };
  in
    {
      devShells.${system} = rec {
        develop = fhs.env;
        default = develop;
      };
    };
}

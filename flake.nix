{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { nixpkgs, ... }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        # CUDA のうち NVRTC だけ unfree を許す
        config.allowUnfreePredicate = p: builtins.elem (nixpkgs.lib.getName p) [ "cuda_nvrtc" ];
      };
      # ドライバ (CUDA 13.3) より新しい NVRTC の PTX は読み込めないので合わせる
      nvrtc = pkgs.cudaPackages_13_3.cuda_nvrtc;
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [ cargo rustc clippy rustfmt rust-analyzer ];
        # libcuda は WSL が /usr/lib/wsl/lib に置く。libnvrtc は nix から
        LD_LIBRARY_PATH = "/usr/lib/wsl/lib:${nvrtc.lib}/lib";
      };
    };
}

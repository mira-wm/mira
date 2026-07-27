#!/usr/bin/env bash
# Download and atomically install the frozen Dust2 codec and single-MIRA baseline.
#
# Required:
#   CS1K_FROZEN_BUNDLE_URL  A current HTTPS URL for the immutable bundle.
#   CS1K_CHECKPOINT_ROOT     Destination containing codec/ and mira-single-step15000/.
set -euo pipefail

bundle_url=${CS1K_FROZEN_BUNDLE_URL:?Set the presigned frozen-endpoint bundle URL}
checkpoint_root=${CS1K_CHECKPOINT_ROOT:?Set the checkpoint destination root}
bundle_sha256=198b53912948a53fd248f68cd7219d0526ab0965ab717eb59e18911c4a3664e2
bundle_path=${CS1K_FROZEN_BUNDLE_PATH:-$checkpoint_root/mira-dust2-frozen-inputs-v1.tar}

for command_name in curl mktemp mv sha256sum tar; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done
mkdir -p "$checkpoint_root"

download_tmp=
staging_dir=
cleanup() {
  if [[ -n "$download_tmp" && -e "$download_tmp" ]]; then
    rm -f -- "$download_tmp"
  fi
  if [[ -n "$staging_dir" && -d "$staging_dir" ]]; then
    rm -rf -- "$staging_dir"
  fi
}
trap cleanup EXIT

if [[ -e "$bundle_path" ]]; then
  if [[ "$(sha256sum "$bundle_path" | cut -d' ' -f1)" != "$bundle_sha256" ]]; then
    echo "Existing bundle has the wrong SHA-256: $bundle_path" >&2
    exit 1
  fi
else
  download_tmp=$(mktemp "$checkpoint_root/.mira-dust2-frozen-inputs.XXXXXX")
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --output "$download_tmp" \
    "$bundle_url"
  if [[ "$(sha256sum "$download_tmp" | cut -d' ' -f1)" != "$bundle_sha256" ]]; then
    echo "Downloaded bundle SHA-256 does not match the frozen contract" >&2
    exit 1
  fi
  mv "$download_tmp" "$bundle_path"
  download_tmp=
fi

expected_members=(
  codec/checkpoint-18000/checkpoint.pth
  codec/codec_config.yaml
  mira-single-step15000/checkpoint.pth
  mira-single-step15000/world_model_config.yaml
)
mapfile -t actual_members < <(tar -tf "$bundle_path" | LC_ALL=C sort)
mapfile -t sorted_expected_members < <(printf '%s\n' "${expected_members[@]}" | LC_ALL=C sort)
if [[ "${actual_members[*]}" != "${sorted_expected_members[*]}" ]]; then
  echo "Bundle member list does not match the frozen contract" >&2
  exit 1
fi

staging_dir=$(mktemp -d "$checkpoint_root/.mira-dust2-stage.XXXXXX")
tar -xf "$bundle_path" -C "$staging_dir"

declare -A expected_sha256=(
  [codec/checkpoint-18000/checkpoint.pth]=3c286c59b74cd141e72af69cde1a0a005142d8d2b472c789cdf2a39a140c4b7a
  [codec/codec_config.yaml]=1e560be4adfbfd9c6864ba0abc9de997e90f2d7156c8a535e4f687fa4c93ba0b
  [mira-single-step15000/checkpoint.pth]=3dbd8f0e43dbe833a5f36370d75f6306c7aa036dfcd3edba767ab138232fa047
  [mira-single-step15000/world_model_config.yaml]=12c6073f745b377392597f8f94d794cf47f876e927506442d4c2cadedae5c492
)

for relative_path in "${expected_members[@]}"; do
  staged_path=$staging_dir/$relative_path
  target_path=$checkpoint_root/$relative_path
  expected=${expected_sha256[$relative_path]}
  if [[ ! -f "$staged_path" ]] ||
    [[ "$(sha256sum "$staged_path" | cut -d' ' -f1)" != "$expected" ]]; then
    echo "Frozen artifact failed verification: $relative_path" >&2
    exit 1
  fi
  if [[ -e "$target_path" ]]; then
    if [[ ! -f "$target_path" ]] ||
      [[ "$(sha256sum "$target_path" | cut -d' ' -f1)" != "$expected" ]]; then
      echo "Refusing to overwrite a conflicting destination: $target_path" >&2
      exit 1
    fi
    continue
  fi
  mkdir -p "$(dirname "$target_path")"
  mv "$staged_path" "$target_path"
done

for relative_path in "${expected_members[@]}"; do
  printf '%s  %s\n' "${expected_sha256[$relative_path]}" "$checkpoint_root/$relative_path"
done
printf 'Frozen Dust2 endpoints staged under %s\n' "$checkpoint_root"

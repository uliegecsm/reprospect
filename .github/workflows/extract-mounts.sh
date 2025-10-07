#!/bin/sh

set -e

MOUNT_PATHS=""

# Loop over the inputs (Docker files).
for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo "'$file' was not found."
        exit -1
    fi

    echo "Processing '$file'."

    # Extract all --mount=type=bind,source=... patterns using POSIX-compliant tools.
    tmp=$(grep -o '\--mount=[^ ]*' "$file" | \
        grep 'type=bind' | \
        sed -n 's/.*source=\([^,]*\).*/\1/p' | \
        sort -u | \
        tr '\n' ' ')

    if [ -n "$tmp" ]; then
        echo "  Found in $file: $tmp"
        MOUNT_PATHS="$MOUNT_PATHS $tmp"
    fi
done

# Remove duplicates and trim whitespace.
MOUNT_PATHS=$(echo "$MOUNT_PATHS" | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/^ *//;s/ *$//')

echo "All mounted paths from $@: $MOUNT_PATHS"

# Output for GitHub Actions.
if [ -n "$GITHUB_OUTPUT" ]; then
    echo "paths=$MOUNT_PATHS" >> "$GITHUB_OUTPUT"
fi

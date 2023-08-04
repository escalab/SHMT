#!/bin/sh

# ===== use pre-compiled shared library =====
sudo cp ../lib/aarch64/libgptpu_utils.so /usr/local/lib/
sudo cp ../lib/aarch64/gptpu_utils.h /usr/local/include/

# ===== if compile from edgetpu source =====
#sudo cp ../edgetpu/out/aarch64/examples/libgptpu_utils.so /usr/local/lib/
#sudo cp ../edgetpu/src/cpp/examples/gptpu_utils.h /usr/local/include/

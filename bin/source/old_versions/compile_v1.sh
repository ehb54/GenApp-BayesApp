version=15
gfortran iftci_v$version.f -march=native -O2 -o iftci
cp iftci_v$version.f iftci_latest_version.f

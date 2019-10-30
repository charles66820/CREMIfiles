unzip source-archive.zip
cd gamefromrussia/trunk/
mkdir build
cd build
cmake ..
make
./thatgamefromrussia

cd ../../../
read -r -p "${1:-Remove gamefromrussia? [y/N]} " response
case "$response" in
    [yY][eE][sS]|[yY])
        rm -rf gamefromrussia
        ;;
esac
echo -e "Done."

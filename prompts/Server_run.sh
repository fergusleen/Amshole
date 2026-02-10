docker run -d \
  --name telstar-server \
  --network telstar-network \
  -p 6512:6502 \
  -e TELSTAR_START_PAGE=100 \
  -e TELSTAR_MAIN_INDEX_PAGE=100 \
  -e TELSTAR_DBCOLLECTION=SECONDARY \
   -e TELSTAR_SERVER_DISPLAY_NAME=AMSHOLE \
  -v telstar-volume:/opt/telstar/volume \
  johnnewcombe/telstar:arm64v8-2.0.29-rc1 server --port=6502
Magic Card Detector
-------------------

## Getting Started
```
mdir build
cd build
cmake ../
make
```

## Usage
Detect a card and save it to the current directory
```
./carddetector --image path/to/image.png --save
```

Just detect a card
```
./carddetector --image path/to/image.png
```

Use the webcam instead of an image
```
./carddetector --webcam
```

# OBJECT-02: THE ROADTRIP

Go on a virtual roadtrip anywhere you want! Enter your route and imagine you're sitting in the back seat, staring out the window.

I have been uploading some updates on [Twitter](https://twitter.com/jordan_armitage)


## REQUIREMENTS

- Get a Google API key [here](https://developers.google.com/maps/get-started)
- Turn on the Google Maps API, specifically the Directions API and the Street View Static API.


## HOW IT WORKS

1. Use [Google Maps](https://www.google.com/maps) to create your route. Add waypoints and change the route by dragging the line on Google Maps.
2. Copy the whole URL (https://www.google.com/maps/dir/...) and paste it where it says "Please enter a Google Maps URL: "
3. Double check the route details (start, destination, and any waypoints). If it looks good, type 'Y' or 'y' and hit enter.
4. Images will download and save in a folder, and your video will generate automatically.

Here are some modifications that you can make:
- You'll look out the right side of the car by default. To look out the left side, change the WINDOW_ANGLE from 90 to 270.
- Play with the FPS (frames per second) to "drive" faster or slower.
- Adjust the RADIUS to set the max distance between street view images. If you want fewer images for less API cost or faster processing, increase this number.


## IMPROVEMENTS

Here are some of the next steps that I'm working on:

- [x] Add a filter for images that don't belong, possibly by tweaking the similarity check.
- [ ] Add a Google Maps overlay with the full route and current location. It should help when you get lost.
- [ ] Add a car door overlay so that it feels more like looking out a car window.
- [ ] Improve image transition - motion blur? inpainting?
- [ ] Create a web application frontend to add a UI

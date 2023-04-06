import re
import requests
import polyline
import math
import os
from PIL import Image, ImageChops
from io import BytesIO
import numpy as np
import cv2
import glob

# GOOGLE API
API_KEY = 'API_KEY'
api_counter = 0

# VARIABLES
WINDOW_ANGLE = 90       # 90deg to look out the right window, 270deg for left
FPS = 6                 # Video frames per second
RADIUS = 6              # Interpoaltion distance

FOV_INCREASE = 15       # Angle added in front or behind

def decode_google_maps_url(url):
    pattern = r"https://www\.google\.com/maps/dir/(.*)/(.*)/@"
    matches = re.findall(pattern, url)
    start_address = matches[0][0].replace("+", " ")
    destination_address = matches[0][1].replace("+", " ")
    
    waypoint_pattern = r"!3m4!1m2!1d(.*?)!2d(.*?)!3s"
    waypoints = re.findall(waypoint_pattern, url)

    decoded = {
        "start_address": start_address,
        "destination_address": destination_address,
        "waypoints": [(float(waypoint[1]), float(waypoint[0])) for waypoint in waypoints]
    }
    return decoded

def get_directions(start_address, destination_address, waypoints):
    global api_counter

    waypoints_str = '|'.join([f'via:{lat},{lng}' for lat, lng in waypoints])

    url = f'https://maps.googleapis.com/maps/api/directions/json?' \
          f'origin={start_address}' \
          f'&destination={destination_address}' \
          f'&waypoints={waypoints_str}' \
          f'&key={API_KEY}'
    
    response = requests.get(url)
    directions = response.json()
    api_counter += 1

    if directions['status'] == 'OK':
        return directions

    else:
        raise Exception(f'Error: {directions["status"]}')

def get_distance(directions_data):
    route = directions_data["routes"][0]
    legs = route["legs"]

    total_distance = 0
    for leg in legs:
        total_distance += leg["distance"]["value"]

    return total_distance / 1000  # Convert meters to kilometers

def get_points(directions_data): 
    encoded_polyline = directions_data['routes'][0]['overview_polyline']['points']
    points = polyline.decode(encoded_polyline)

    # Interpolate points along the polyline
    interpolated_points = []
    for i in range(len(points) - 1):
        interpolated_points.extend(interpolate_points(points[i], points[i + 1]))

    return interpolated_points
    
def interpolate_points(point1, point2, dist=RADIUS):
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])

    d_lat, d_lon = lat2 - lat1, lon2 - lon1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    R = 6371e3  # Earth's radius in meters
    d = R * c

    interpol_count = int(d // dist)
    if interpol_count == 0:
        return [point1]

    fraction = 1 / interpol_count
    interpolated_points = []

    for i in range(interpol_count + 1):
        f = i * fraction
        a = math.sin((1 - f) * c) / math.sin(c)
        b = math.sin(f * c) / math.sin(c)

        x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
        y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
        z = a * math.sin(lat1) + b * math.sin(lat2)

        lat3 = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
        lon3 = math.atan2(y, x)

        interpolated_points.append((math.degrees(lat3), math.degrees(lon3)))

    return interpolated_points

def get_bearing(lat1, lon1, lat2, lon2):
    R = 6371 # radius of the Earth in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1

    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)

    return bearing

def average_angle(angle1, angle2):
    angle1_rad = math.radians(angle1)
    angle2_rad = math.radians(angle2)

    x1, y1 = math.cos(angle1_rad), math.sin(angle1_rad)
    x2, y2 = math.cos(angle2_rad), math.sin(angle2_rad)

    x_avg, y_avg = (x1 + x2) / 2, (y1 + y2) / 2

    average_angle_rad = math.atan2(y_avg, x_avg)
    average_angle_deg = math.degrees(average_angle_rad)

    if average_angle_deg < 0:
        average_angle_deg += 360

    return average_angle_deg

def is_same_image(image1, image2, threshold=100):
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    mse = np.mean((image1_array - image2_array) ** 2)

    return mse < threshold

def points_to_bearings(points):
    bearings = []
    
    for i in range(len(points)):
        if i == 0:                      # First point
            lat1, lon1 = points[i]
            lat2, lon2 = points[i+1]
            bearing = get_bearing(lat1, lon1, lat2, lon2)
            bearings.append(bearing)
        elif i == len(points) - 1:      # Last point
            lat1, lon1 = points[i-1]
            lat2, lon2 = points[i]
            bearing = get_bearing(lat1, lon1, lat2, lon2)
            bearings.append(bearing)
        else:                           # Everything else
            lat1, lon1 = points[i-1]
            lat2, lon2 = points[i]
            bearing1 = get_bearing(lat1, lon1, lat2, lon2)
            lat1, lon1 = points[i]
            lat2, lon2 = points[i+1]
            bearing2 = get_bearing(lat1, lon1, lat2, lon2)
            avg_bearing = average_angle(bearing1, bearing2)
            bearings.append(avg_bearing)

    return bearings

def get_images(points, bearings, images_folder):
    prev_image = None
    order_number = 0

    for i, (point, bearing) in enumerate(zip(points, bearings)):
        total_images = len(points)
        progress = round(100*i/total_images)
        print(f"\r      Downloading Images: ({i}/{total_images})  {progress}%", end='', flush=True)
        
        latitude, longitude = point
        adjusted_bearing = (bearing + WINDOW_ANGLE) % 360

        image = streetview_api(latitude, longitude,adjusted_bearing)

        # Compare the current image with the previous one
        if prev_image is not None and is_same_image(image, prev_image, threshold=50):
            # If the images are the same, skip saving the image and continue with the next iteration
            continue
        
        else:
            images = []
            images.append(image)
            more_image = streetview_api(latitude, longitude,adjusted_bearing - FOV_INCREASE)
            images.append(more_image)
            more_image = streetview_api(latitude, longitude,adjusted_bearing + FOV_INCREASE)
            images.append(more_image)

            converted_images = []
            for i in range(3):
                converted_image = convert_image(images[i])
                converted_images.append(converted_image)
            
            stitch_result_left = stitch([flip_image(converted_images[0]), flip_image(converted_images[1])])
            stitch_result = stitch([flip_image(stitch_result_left), converted_images[2]])

            stitch_result = convert_image(stitch_result)

            # Save the image to a file with the current order number in the "Images" folder
            filename = os.path.join(images_folder, f"{order_number}.jpg")
            img = Image.fromarray(stitch_result)
            img.save(filename)

            # Update the previous image and increment the order number
            prev_image = image
            order_number += 1

    print("\r      Downloading Images: Done                                  \n", flush=True)

def streetview_api(latitude, longitude, heading):
    global api_counter

    # Set the parameters for the API request
    params = {
        "location": f"{latitude},{longitude}",
        "size": "640x640",
        "fov": "120",
        "heading": str(heading),
        "pitch": "0",
        "radius": RADIUS,
        "key": API_KEY,
    }

    # Make the API request
    endpoint = "https://maps.googleapis.com/maps/api/streetview"
    response = requests.get(endpoint, params=params)
    api_counter += 1

    # Parse the response as an image
    image = Image.open(BytesIO(response.content))
    return image   

def convert_image(image):
    image_np = np.array(image)
    converted_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return converted_image

def flip_image(image):
    flipped_image = cv2.flip(image, 1)
    return flipped_image

def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    (imageB, imageA) = images
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    if M is None:
        return None

    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H,
                                    (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    if showMatches:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)

    return result

def detectAndDescribe(image):  
    descriptor = cv2.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)

    return None

def create_video_from_images(output_video, folder):
    img_files = glob.glob(os.path.join(folder, '*.jpg'))
    img_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    if not img_files:
        print("      No images found in the folder.")
        return

    first_img = cv2.imread(img_files[0])
    height, width, _ = first_img.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))

    for img_file in img_files:
        img = cv2.imread(img_file)
        video.write(img)

    video.release()
    print(f"      Video created")

if __name__ == "__main__":
    input_url = input("\n\n      Please enter a Google Maps URL: ")
    url_decoded = decode_google_maps_url(input_url)

    print("\n      Start Address:", url_decoded["start_address"])
    print("      Destination Address:", url_decoded["destination_address"])
    print("      Waypoints:", url_decoded["waypoints"])

    directions_data = get_directions(url_decoded["start_address"], url_decoded["destination_address"], url_decoded["waypoints"])
    
    distance = get_distance(directions_data)
    print("      Total Distance: ", distance, " km")
    
    points = get_points(directions_data)
    cost = round( ((len(points)*3) + 5) * 0.007, 2)
    # print("\n      Cost to run: (up to) ", cost, "$")

    decision = input("      Press 'Y' or 'y' to continue, any other key to cancel: ")
    if decision.lower() == 'y':
        print("")
        folder_name = url_decoded["start_address"] + " to " + url_decoded["destination_address"]
        os.makedirs(folder_name)
        images_folder = os.path.join(folder_name, "images")
        os.makedirs(images_folder)

        bearings = points_to_bearings(points)

        get_images(points, bearings, images_folder)
        
        video_name = folder_name + '.mp4'
        output_video = os.path.join(folder_name,video_name)
        create_video_from_images(output_video, images_folder)

    else:
        print("\n      Cancelled.")

    actual_cost = round(api_counter* 0.007, 2)
    # print("\n      Actual Cost: ", actual_cost, "$")

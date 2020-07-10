
class DataSequence(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            self.coords = np.zeros((sum(1 for line in file), 4))
            file.seek(0)

            reader = csv.reader(file, delimiter=",") # Read CSV file
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:5]): # Parse row with seven entities
                    row[i+1] = int(r)

                path,  x0, y0, width,height, _ = row # Read image, its dimensions, BBox coords
                self.coords[index, 0] = x0 * IMAGE_SIZE / ORIGINAL_IMAGE_SIZE # Normalize bounding box by image size
                self.coords[index, 1] = y0 * IMAGE_SIZE / ORIGINAL_IMAGE_SIZE # Normalize bounding box by image size
                self.coords[index, 2] = width * IMAGE_SIZE / ORIGINAL_IMAGE_SIZE # Normalize bounding box by image size
                self.coords[index, 3] = height * IMAGE_SIZE / ORIGINAL_IMAGE_SIZE # Normalize bounding box by image size

                self.paths.append(DATASET_FOLDER+path+'.dcm') # Read image from here

    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, idx): # Get a batch
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE] # Image path
        batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE] # Image coords

        batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            image = pydicom.dcmread(f) # Read image
            image = image.pixel_array
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image = image.reshape( IMAGE_SIZE, IMAGE_SIZE,1)

            batch_images[i] = preprocess_input(np.array(image, dtype=np.float32)) # Convert to float32 array
            

        return batch_images, batch_coords
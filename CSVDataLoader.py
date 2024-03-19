class CSVDataLoader:
    def __init__(self, file_path, batch_size, shuffle=True, peek=False, verbose=False, subset=False, subset_samples=5000, prediction_loader=False):
        # Read all passed parameters
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.peek = peek
        self.verbose = verbose
        self.subset = subset
        self.subset_samples = subset_samples
        self.prediction_loader = prediction_loader  # to avoid infinite prediction

        # Set num_samples according to subset logic
        total_samples = count_samples_in_csv(file_path)
        if self.subset:
            if total_samples <= self.subset_samples:
                print(f"Number of samples in the data is less than or equal to the subset size. No subset will be created.")
                self.num_samples = total_samples
            else:
                print(f"Using a subset of the data with {subset_samples} samples.")
                self.num_samples = self.subset_samples
        else:
            self.num_samples = total_samples
        
        # Set chunk size equal to num_samples, or a multiple of batch size if less than the number of samples
        self.chunk_size = min(int(np.ceil(30000 / self.batch_size)) * self.batch_size, self.num_samples)

        # Set indices and steps
        self.indices = np.arange(1, self.num_samples + 1)  # Add one to skip the header
        self.steps = int(np.ceil(self.num_samples / self.batch_size))

        # Shuffle if set
        if shuffle:
            np.random.shuffle(self.indices)

        # Initializations
        self.current_index = 0  # this is the index for the total dataset indices
        self.chunk_index = 0  # this is the index inside the data chunk
        self.initialize_chunk = True
        self.one_chunk = self.chunk_size == self.num_samples  # determine if the dataset size is just one chunk
        self.peek_show = True
        self.chunk_data = None

    def __iter__(self):
        if self.verbose:
            print("Data Loader Started")
        return self

    def __next__(self):
        """Read a chunk at a time from the CSV file and yield batches from it. This allows us to search the dataframe (expensive)
        once per chunk, rather than once per batch."""

        # Load first chunk
        if self.initialize_chunk == True:
            self.load_chunk()
            self.initialize_chunk = False

        """Epoch"""
        # Set state at the start of each epoch
        if self.current_index >= self.num_samples:  # Start new epoch
            if self.prediction_loader:
                raise StopIteration  # To avoid infinite prediction
            if self.verbose:
                print("\n-------------------NEW EPOCH-------------------")
            # Reset indices
            self.current_index = 0
            self.chunk_index = 0
            # Shuffle if set
            if self.shuffle:
                if self.verbose:
                    print("\n-------------------SHUFFLING-------------------")
                if self.one_chunk:
                    self.chunk_data = self.chunk_data.sample(frac=1).reset_index(drop=True)
                else:
                    np.random.shuffle(self.indices)
                    self.load_chunk()
            

        """Chunk"""
        if self.chunk_index >= self.chunk_size and not self.one_chunk:
            self.load_chunk()
            self.chunk_index = 0

        """Batch"""
        # Read batch data from chunk
        batch_start = self.chunk_index
        batch_end = min(batch_start + self.batch_size, self.chunk_size)
        batch_data = self.chunk_data[batch_start:batch_end]

        # Extract features and labels from batch_data
        batch_x = batch_data.iloc[:, 1:].to_numpy()
        batch_y = batch_data.iloc[:, 0].to_numpy()

        # Update indices if not peeking
        if self.peek:
            if self.verbose:
                print("Peeking at batch data")
                if self.peek_show:
                    print("X Batch Data:")
                    batch_x_df = pd.DataFrame(batch_x)
                    print(batch_x_df.head())
                    print(batch_x_df.shape)
                    print("Y Batch Data:")
                    batch_y_df = pd.DataFrame(batch_y)
                    print(batch_y_df.head())
                    self.peek_show = False
            self.peek = False
        else:
            self.current_index += self.batch_size  # Update the current index for the next call
            self.chunk_index += self.batch_size  # Update the chunk index for the next call

        return batch_x, batch_y

    def load_chunk(self):
        if self.verbose:
            print("\n-------------------NEW CHUNK-------------------")
        chunk_start = self.current_index
        chunk_end = min(chunk_start + self.chunk_size, self.num_samples)
        not_chunk_indices = np.concatenate([self.indices[:chunk_start], self.indices[chunk_end:]])  # Indices not in chunk
        self.chunk_data = pd.read_csv(self.file_path, skiprows=not_chunk_indices, nrows=self.chunk_size)

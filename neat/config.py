class NEATConfig:
    """
    Configuration class for the algorithm.

    Stores various hyperparameters and settings that control the evolution process,
    genome structure, mutation rates and speciation.
    """
    def __init__(
            self,
            population_size=100,
            genome_shape=(1, 1),
            hid_node_activation='relu',
            out_node_activation='sigmoid',
            max_weight=1.0,
            min_weight=-1.0,
            min_bias=-1.0,
            max_bias=1.0,
            add_node_mutation_prob=0.05,
            add_conn_mutation_prob=0.08,
            remove_conn_mutation_prob=0.02,
            remove_node_mutation_prob=0.01,
            num_elites=1,
            selection_share=0.2,
            sigma=0.1,
            perturb_prob=0.8,
            reset_prob=0.1,
            species_threshold=3.0,
            min_species_threshold=0.15,
            max_species_threshold=15.0,
            target_species_number=15,
            adaptive_threshold=0.0,
            c1=1.0,
            c2=1.0,
            c3=0.4,
            save_path="./",
    ):
        """
        Initializes the NEATConfig object with specified or default parameters.
        """
        self.genome_shape = genome_shape
        self.add_node_mutation_prob = add_node_mutation_prob
        self.add_conn_mutation_prob = add_conn_mutation_prob
        self.remove_node_mutation_prob = remove_node_mutation_prob
        self.remove_conn_mutation_prob = remove_conn_mutation_prob
        self.sigma = sigma
        self.perturb_prob = perturb_prob
        self.reset_prob = reset_prob
        self.hid_node_activation = hid_node_activation
        self.out_node_activation = out_node_activation
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.min_bias = min_bias
        self.max_bias = max_bias
        self.species_threshold = species_threshold
        self.min_species_threshold = min_species_threshold
        self.max_species_threshold = max_species_threshold
        self.adaptive_threshold = adaptive_threshold
        self.target_species_number = target_species_number
        self.population_size = population_size
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_elites = num_elites
        self.selection_share = selection_share
        self.save_path = save_path

    def __repr__(self):
        """
        Returns a string representation of the configuration settings.
        """
        str_repr = ""
        for key, value in self.__dict__.items():
            str_repr += f"{key}: {value}\n"
        return str_repr

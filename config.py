class config:
    
    # generation of the compressed dataset
    image_size = 512
    dataset_name = "huggan/wikiart"
    stable_diffusion_id = "stabilityai/stable-diffusion-2-1-base"
    vae_scale = 0.18215
    dataset_dir = "./dataset"

    # compressed dataset
    dataset_path = 'dataset/compressed_wikiart'
    
    artists = ['Unknown Artist', 'boris-kustodiev', 'camille-pissarro', 'childe-hassam', 
               'claude-monet', 'edgar-degas', 'eugene-boudin', 'gustave-dore', 
               'ilya-repin', 'ivan-aivazovsky', 'ivan-shishkin', 'john-singer-sargent', 
               'marc-chagall', 'martiros-saryan', 'nicholas-roerich', 'pablo-picasso', 
               'paul-cezanne', 'pierre-auguste-renoir', 'pyotr-konchalovsky', 'raphael-kirchner', 
               'rembrandt', 'salvador-dali', 'vincent-van-gogh', 'hieronymus-bosch',
               'leonardo-da-vinci', 'albrecht-durer', 'edouard-cortes', 'sam-francis', 
               'juan-gris', 'lucas-cranach-the-elder', 'paul-gauguin', 'konstantin-makovsky', 
               'egon-schiele', 'thomas-eakins', 'gustave-moreau', 'francisco-goya',
               'edvard-munch', 'henri-matisse', 'fra-angelico', 'maxime-maufra', 
               'jan-matejko', 'mstislav-dobuzhinsky', 'alfred-sisley', 'mary-cassatt', 
               'gustave-loiseau', 'fernando-botero', 'zinaida-serebriakova', 'georges-seurat',
               'isaac-levitan', 'joaquã\xadn-sorolla', 'jacek-malczewski', 'berthe-morisot', 
               'andy-warhol', 'arkhip-kuindzhi', 'niko-pirosmani', 'james-tissot', 
               'vasily-polenov', 'valentin-serov', 'pietro-perugino', 'pierre-bonnard',
               'ferdinand-hodler', 'bartolome-esteban-murillo', 'giovanni-boldini', 'henri-martin', 
               'gustav-klimt', 'vasily-perov', 'odilon-redon', 'tintoretto', 
               'gene-davis', 'raphael', 'john-henry-twachtman', 'henri-de-toulouse-lautrec', 
               'antoine-blanchard', 'david-burliuk', 'camille-corot', 'konstantin-korovin',
               'ivan-bilibin', 'titian', 'maurice-prendergast', 'edouard-manet', 
               'peter-paul-rubens', 'aubrey-beardsley', 'paolo-veronese', 'joshua-reynolds', 
               'kuzma-petrov-vodkin', 'gustave-caillebotte', 'lucian-freud', 'michelangelo', 
               'dante-gabriel-rossetti', 'felix-vallotton', 'nikolay-bogdanov-belsky', 'georges-braque', 
               'vasily-surikov', 'fernand-leger', 'konstantin-somov', 'katsushika-hokusai', 
               'sir-lawrence-alma-tadema', 'vasily-vereshchagin', 'ernst-ludwig-kirchner', 'mikhail-vrubel', 
               'orest-kiprensky', 'william-merritt-chase', 'aleksey-savrasov', 'hans-memling', 
               'amedeo-modigliani', 'ivan-kramskoy', 'utagawa-kuniyoshi', 'gustave-courbet',
               'william-turner', 'theo-van-rysselberghe', 'joseph-wright', 'edward-burne-jones', 
               'koloman-moser', 'viktor-vasnetsov', 'anthony-van-dyck', 'raoul-dufy', 
               'frans-hals', 'hans-holbein-the-younger', 'ilya-mashkov', 'henri-fantin-latour', 
               'm.c.-escher', 'el-greco', 'mikalojus-ciurlionis', 'james-mcneill-whistler', 
               'karl-bryullov', 'jacob-jordaens', 'thomas-gainsborough','eugene-delacroix', 'canaletto']

    genre = ['abstract_painting', 'cityscape', 'genre_painting', 'illustration',
             'landscape', 'nude_painting', 'portrait', 'religious_painting',
             'sketch_and_study', 'still_life', 'Unknown Genre']

    style = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau', 
             'Baroque', 'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 
             'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance', 
             'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism', 
             'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art', 
             'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 
             'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']

    # model
    pretrained_path = 'latent-ddpm-wiki-art-512-64-run2/unet'
    in_channels = 4
    out_channels = 4
    sample_size = 64
    layers_per_block = 2
    block_out_channels = (128, 256, 512, 512)
    down_block_types = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    up_block_types = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    num_artist_classes = 129
    num_genre_classes = 11
    num_style_classes = 27
    class_embed_dropout = 0.1

    # training
    train_batch_size = 5
    eval_batch_size = 8

    num_workers = 2
    num_epochs = 250
    gradient_accumulation_steps = 1
    learning_rate = 2.5e-5
    weight_decay = 1e-6
    adam_beta1 = 0.95
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    lr_warmup_steps = 500
    save_image_steps = -1
    save_model_steps = 1000
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'latent-ddpm-wiki-art-512-64-run2'
    
    use_ema = True
    ema_inv_gamma = 1.0
    ema_power = 3/4
    ema_max_decay = 0.9999

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 42

    # Evaluation
    evaluation_artists = ['salvador-dali', 'ivan-aivazovsky', 'martiros-saryan', 'vincent-van-gogh', 
                          'nicholas-roerich', 'pierre-auguste-renoir', 'claude-monet', 'pyotr-konchalovsky']

    evaluation_genres = ['religious_painting', 'Unknown Genre', 'landscape', 'sketch_and_study', 
                         'landscape', 'portrait', 'landscape', 'landscape']

    evaluation_styles = ['Expressionism', 'Romanticism', 'Expressionism', 'Post_Impressionism', 
                         'Symbolism', 'Impressionism', 'Impressionism', 'Post_Impressionism']

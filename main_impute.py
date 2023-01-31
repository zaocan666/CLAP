from run_script.run_epochs_vaes import run_epochs_vae
from utils.main_comp import main_start, main_end


if __name__ == '__main__':
    dataset = 'polymnist'
    if dataset == 'mnistsvhntext':
        from mnistsvhntext.flags import parser as mnistsvhntext_parser
        parser = mnistsvhntext_parser
    elif dataset == 'polymnist':
        from mmnist.flags import parser as mnist_parser
        parser = mnist_parser
    elif dataset == 'celeba':
        from celeba.flags import parser as celeba_parser
        parser = celeba_parser
    elif dataset == 'eicu':
        from eicu.flags import parser as eicu_parser
        parser = eicu_parser

    FLAGS = parser.parse_args()
    assert FLAGS.dataset==dataset
    FLAGS, mst = main_start(FLAGS)
    run_epochs_vae(mst)

    main_end(FLAGS)

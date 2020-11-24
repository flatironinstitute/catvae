import argparse
from biom import load_table


def add_model_specific_args(parent_parser, add_help=True):
    parser = argparse.ArgumentParser(parents=[parent_parser],
                                     add_help=add_help)
    parser.add_argument(
        '--biom-file', help='Biom file to convert to CTM friendly format.',
        required=True)
    parser.add_argument(
        '--output-file', help='Output CTM file.', required=True)
    return parser


def biom2ctm(file_name, output_name):
    table = load_table(file_name)
    species = table.ids(axis='observation')
    matrix = table.matrix_data.T.tolil()
    with open(output_name, 'w') as fh:
        for i in range(len(matrix.rows)):
            ids = matrix.rows[i]
            counts = map(str, map(int, matrix.data[i]))
            names = map(str, species[ids])
            M = len(species)
            res = zip(names, counts)
            res = list(map(lambda x: ':'.join(x), res))
            line = str(M) + ' ' + ' '.join(res) + '\n'
            fh.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = add_model_specific_args(parser)
    args = parser.parse_args()
    biom2ctm(args.biom_file, args.output_file)

'''
Script to parse data from dnd

dialogue history string --> dialogue act history string

RUN FROM "src/data/cocoa/scripts"
'''

import argparse

# <input> 1 4 4 1 1 2 </input>
# <dialogue> THEM: i would like 4 hats and you can have the rest . <eos> YOU: deal <eos> THEM: <selection> </dialogue> 
# <output> item0=1 item1=0 item2=1 item0=0 item1=4 item2=0 </output> 
# <partner_input> 1 0 4 2 1 2 </partner_input>


def parse_inp(scn: str):
    # takes in like this: "1 4 4 1 1 2"
    
    # TODO
    pass


def parse_dialogue(dia: str):
    # takes in like this: "THEM: i would like 4 hats and you can have the rest . <eos> YOU: deal <eos> THEM: <selection>"
    # or this: ""
    
    # TODO
    pass


def parse_output(out: str):
    # takes in like this: "item0=1 item1=0 item2=1 item0=0 item1=4 item2=0"
    # or this: "<no_agreement> <no_agreement> <no_agreement> <no_agreement> <no_agreement> <no_agreement>"
    
    # TODO
    pass


def parse_line(l: str):
    sp = l.split(' </input> ')
    assert(len(sp) == 2)
    inp = parse_inp(sp[0].lstrip('<input> '))

    sp = sp[1].split(' </dialogue> ')
    assert(len(sp) == 2)
    dia = parse_dialogue(sp[0].lstrip('<dialogue> '))

    sp = sp[1].split(' </output> ')
    assert(len(sp) == 2)
    out = parse_dialogue(sp[0].lstrip('<output> '))

    part_inp = parse_inp(sp[1].lstrip('<partner_input> ').rstrip(' </partner_input>'))

    # TODO: DO STUFF WITH inp, dia, out, part_inp


def parse_file(f, out):
    line = f.readline()
    while line != '':
        p_line = parse_line(line)
        out.write(p_line)

        line = f.readline()


def main():
    parser = argparse.ArgumentParser(description='dialogue history string --> dialogue act history string')
    parser.add_argument('--train_file', type=str, default='../../negotiate/train.txt',
        help='location of the unparsed train file')
    parser.add_argument('--val_file', type=str, default='../../negotiate/val.txt',
        help='ocation of the unparsed val file')
    parser.add_argument('--test_file', type=str, default='../../negotiate/test.txt',
        help='ocation of the unparsed test file')
    args = parser.parse_args()

    # Parse files
    with open(args.train_file, 'r') as in_f, open('../train.txt', 'w') as out_f:
        parse_file(in_f, out_f)

    with open(args.val_file, 'r') as in_f, open('../val.txt', 'w') as out_f:
        parse_file(in_f, out_f)

    with open(args.test_file, 'r') as in_f, open('../test.txt', 'w') as out_f:
        parse_file(in_f, out_f)


if __name__ == '__main__':
    main()

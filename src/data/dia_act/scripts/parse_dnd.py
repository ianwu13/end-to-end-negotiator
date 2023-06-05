'''
Script to parse data from dnd

dialogue history string --> dialogue act history string

RUN FROM "src/data/cocoa/scripts"
'''

import argparse

from prop_extract import *

# <input> 1 4 4 1 1 2 </input>
# <dialogue> THEM: i would like 4 hats and you can have the rest . <eos> YOU: deal <eos> THEM: <selection> </dialogue> 
# <output> item0=1 item1=0 item2=1 item0=0 item1=4 item2=0 </output> 
# <partner_input> 1 0 4 2 1 2 </partner_input>


def parse_inp(scn: str):
    # takes in like this: "1 4 4 1 1 2"
    
    # TODO
    return scn


def parse_dialogue(dia: str):
    # takes in like this: "THEM: i would like 4 hats and you can have the rest . <eos> YOU: deal <eos> THEM: <selection>"
    # or this: ""

    # TODO
    if dia[0] == 'T':
        agent = 0
    elif dia[0] == 'Y':
        agent = 1
    else:
        raise Exception(f'Agent cannot be identified from dialogue:\n\n{dia}\n\n')

    ex = get_ex(dia)
    if ex is None:
        return None
    ex = AttributeDict(ex)

    dialogue = Dialogue(agent, ex.scenario['kbs'][agent],
                    ex.outcome, ex.uuid, model='lf2lf')
    for e in ex.events:
        e = AttributeDict(e)
        lf = AttributeDict(e.metadata)
        proposal = None
        assert lf is not None
        # TODO: hack
        if lf.get('proposal') is not None:
            proposal = {'me': {}, 'you': {}}
            # Parser is from the receiver's perspective
            received_proposal = {int(k): v for k, v in lf['proposal'].items()}
            proposal['me'] = received_proposal[1]  # received_proposal[dialogue.agent]
            proposal['you'] = received_proposal[0]  # received_proposal[1-dialogue.agent]
        if e.action == 'select':
            if e.agent != dialogue.agent:
                proposal = None
            else:
                sel = ex.outcome['item_split']
                proposal = {'me': {}, 'you': {}}
                for item, count in sel[dialogue.agent].items():
                    proposal['me'][item] = count
                for item, count in sel[1-dialogue.agent].items():
                    proposal['you'][item] = count
                    
        utterance = dialogue.lf_to_tokens(lf, proposal, items=["book", "hat", "ball"])
            
        if utterance:
            dialogue.add_utterance(e.agent, utterance, lf=e.metadata)
    
    # print(dialogue)
    # print(dia.split('<eos>'))
    # print(dialogue.token_turns, '\n')
    # print(dialogue.lfs, '\n')
    # print(dialogue.turns, '\n')
    # print(dialogue.entities, '\n')
    # print(dialogue.agents, '\n')

    # OTHER TEST
    # print(ex.events)
    # print(dia.split('<eos>'))
    # print([' '.join(u) for u in dialogue.token_turns])
    # raise Exception('DONE')

    # Readjust dialogue token array
    tok_turns = dialogue.token_turns
    if len(tok_turns[0]) == 2:
        tok_turns = tok_turns[1:]
    for turn in tok_turns:
        if agent == 1:
            turn[0] = 'YOU:'
            agent = 0
        else:
            turn[0] = 'THEM:'
            agent = 1

    '''
    for i in [-1, -2]:
        assert(tok_turns[i][1] == '<select>')
        tok_turns[i][1] = 'select'
        tok_turns[i][-1] = '<selection>'

    return ' '.join([' '.join(utt) for utt in tok_turns])
    '''
    
    assert(tok_turns[-2][1] == '<select>')
    tok_turns[-2] = [tok_turns[-2][0], '<selection>']

    return ' '.join([' '.join(utt) for utt in tok_turns[:-1]])


def parse_output(out: str):
    # takes in like this: "item0=1 item1=0 item2=1 item0=0 item1=4 item2=0"
    # or this: "<no_agreement> <no_agreement> <no_agreement> <no_agreement> <no_agreement> <no_agreement>"
    
    # TODO
    return out


def parse_line(l: str):
    sp = l.split(' </input> ')
    assert(len(sp) == 2)
    inp = ''.join([sp[0], ' </input>'])  # parse_inp(sp[0].lstrip('<input> '))

    sp = sp[1].split(' </dialogue> ')
    assert(len(sp) == 2)
    par_d = parse_dialogue(sp[0].lstrip('<dialogue> '))
    if par_d is None:
        return None
    dia = ' '.join(['<dialogue>', par_d, '</dialogue>'])

    sp = sp[1].split(' </output> ')
    assert(len(sp) == 2)
    out = ''.join([sp[0], ' </output>'])  # parse_output(sp[0].lstrip('<output> '))
    if '<no_agreement>' in out:
        return None

    part_inp = sp[1] #  parse_inp(sp[1].lstrip('<partner_input> ').rstrip(' </partner_input>'))
    
    return ' '.join([inp, dia, out, part_inp])


def parse_file(f, out):
    line = f.readline()
    while line != '':
        p_line = parse_line(line)
        if p_line is not None:
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

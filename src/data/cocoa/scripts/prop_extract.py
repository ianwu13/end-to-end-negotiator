from collections import namedtuple, defaultdict
import json


class CanonicalEntity(namedtuple('CanonicalEntity', ['value', 'type'])):
    __slots__ = ()

    def __str__(self):
        return '[%s]' % str(self.value)


class Entity(namedtuple('Entity', ['surface', 'canonical'])):
    __slots__ = ()

    @classmethod
    def from_elements(cls, surface=None, value=None, type=None):
        if value is None:
            value = surface
        return super(cls, Entity).__new__(cls, surface, CanonicalEntity(value, type))

    def __str__(self):
        return '[%s|%s]' % (str(self.surface), str(self.canonical.value))


def is_entity(x):
    return isinstance(x, Entity) or isinstance(x, CanonicalEntity)



# Facebook Negotiation
class Marker:
    EOS = '<eos>'
    PAD = '<pad>'
    
    # Sequence
    GO = '<go>'

    # Actions
    SELECT = '<select>'
    # OFFER = '<offer>'
    # ACCEPT = '<accept>'
    # REJECT = '<reject>'
    QUIT = '<quit>'


markers = Marker

utt_lookup_map = defaultdict(lambda: None)
for f in ['train-parsed.json', 'val-parsed.json', 'test-parsed.json']:
    for dia in json.load(open(f, 'r')):
        utt_lookup_map[''.join([e['data'] for e in dia['events'][:2]])] = dia


def get_ex(dia: str):
    key = ''.join([d.lstrip('THEM: ').lstrip('YOU: ') for d in dia.split(' <eos> ')[:2]])

    res = utt_lookup_map[key]
    if res is None:
        return None
        # raise Exception("res is None")
    '''
    else:
        print()
        print(dia)
        print('8'*100)
        print(' | '.join([e['data'] if type(e['data'])==str else 'SELECTION' for e in res['events']]))
        print()
    '''

    return res


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target

    def __init__(self, agent, kb, outcome, uuid, model='seq2seq'):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        self.model = model
        self.scenario = self.process_scenario(kb)
        self.selection = self.process_selection(outcome['item_split'])
        # token_turns: tokens and entitys (output of entity linking)
        self.token_turns = []
        # parsed logical forms
        self.lfs = []
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = [[], [], []]
        # entities: has the same structure as turns, non-entity tokens are None
        self.entities = []
        self.agents = []
        self.is_int = False  # Whether we've converted it to integers
        self.num_context = None

    @property
    def num_turns(self):
        return len(self.turns[0])

    def join_turns(self):
        for i, utterances in enumerate(self.turns):
            self.turns[i] = [x for utterance in utterances for x in utterance]

    def num_tokens(self):
        return sum([len(t) for t in self.token_turns])

    def add_utterance(self, agent, utterance, lf=None):
        # Always start from the partner agent
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
        self._add_utterance(agent, utterance, lf=lf)

    def _add_utterance(self, agent, utterance, lf=None):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            new_turn = False
        else:
            new_turn = True

        utterance = self._insert_markers(agent, utterance, new_turn)
        entities = [x if is_entity(x) else None for x in utterance]
        if lf:
            lf = self._insert_markers(agent, self.lf_to_tokens(lf), new_turn)
        else:
            lf = []

        if new_turn:
            self.agents.append(agent)

            self.token_turns.append(utterance)
            self.entities.append(entities)
            self.lfs.append(lf)
        else:
            self.token_turns[-1].extend(utterance)
            self.entities[-1].extend(entities)
            self.lfs[-1].extend(lf)

    def lf_to_tokens(self, lf, proposal=None, items=None):
        intent = lf['intent']
        if intent == 'select':
            intent = markers.SELECT
        elif intent == 'quit':
            intent = markers.QUIT
        tokens = [intent]
        if proposal is not None:
            for item in items:
                tokens.append('{item} = {count}'.format(item=item, count= proposal['me'][item]))
                #tokens.append('{count}'.format(count= proposal['me'][item]))
                #tokens.append(str(proposal[item]))
        return tokens

    def _insert_markers(self, agent, utterance, new_turn):
        ''' Add start of sentence and end of sentence markers, ignore other
        markers which were specific to craigslist'''
        utterance.append(markers.EOS)

        if new_turn:
            utterance.insert(0, markers.GO)

        return utterance

    def scenario_to_int(self):
        self.scenario = map(self.mappings['kb_vocab'].to_ind, self.scenario)

    def process_scenario(self, kb):
        attributes = ("Count", "Value")  # "Name"
        scenario = ['{item}-{attr}-{value}'.format(item=fact['Name'], attr=attr, value=fact[attr])
            for fact in kb for attr in attributes]
        assert(len(scenario) == 6)
        return scenario

    def selection_to_int(self):
        # TODO: have a different vocab
        self.selection = map(self.mappings['utterance_vocab'].to_ind, self.selection)

    def process_selection(self, item_split):
        selection = []
        for agent, agent_name in zip((self.agent, 1-self.agent), ('my', 'your')):
            for item in ("book", "hat", "ball"):
                selection.append('{item}={count}'.format(
                    item=item, count=item_split[agent][item]))
        assert(len(selection) == 6)
        return selection

    def lf_to_int(self):
        self.lf_token_turns = []
        for i, lf in enumerate(self.lfs):
            self.lf_token_turns.append(lf)
            self.lfs[i] = map(self.mappings['lf_vocab'].to_ind, lf)

    def convert_to_int(self):
        if self.is_int:
            return

        for turn in self.token_turns:
            # turn is a list of tokens that an agent spoke on their turn
            # self.turns starts out as [[], [], []], so
            #   each portion is a list holding the tokens of either the
            #   encoding portion, decoding portion, or the target portion
            for portion, stage in izip(self.turns, ('encoding', 'decoding', 'target')):
                portion.append(self.textint_map.text_to_int(turn, stage))

        self.scenario_to_int()
        self.selection_to_int()

        self.is_int = True

    def _pad_list(self, l, size, pad):
        for i in xrange(len(l), size):
            l.append(pad)
        return l

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        self.agents = self._pad_list(self.agents, num_turns, None)
        for turns in self.turns:
            self._pad_list(turns, num_turns, [])
        self.lfs = self._pad_list(self.lfs, num_turns, [])


'''
# Takes in metadata fron json
def lf_to_tokens(lf, proposal=None, items=None):
    intent = lf['intent']
    if intent == 'select':
        intent = markers.SELECT
    elif intent == 'quit':
        intent = markers.QUIT
    tokens = [intent]
    if proposal is not None:
        for item in items:
            tokens.append('{item}={count}'.format(item=item, count= proposal['me'][item]))
            #tokens.append('{count}'.format(count= proposal['me'][item]))
            #tokens.append(str(proposal[item]))
    return tokens


def add_utterance(self, agent, utterance, lf=None):
    # Always start from the partner agent
    if len(self.agents) == 0 and agent == self.agent:
        self._add_utterance(1 - self.agent, [], lf={'intent': 'start'})

    # Same agent talking
    if len(self.agents) > 0 and agent == self.agents[-1]:
        new_turn = False
    else:
        new_turn = True

    utterance = self._insert_markers(agent, utterance, new_turn)
    entities = [x if is_entity(x) else None for x in utterance]
    if lf:
        lf = self._insert_markers(agent, self.lf_to_tokens(lf), new_turn)
    else:
        lf = []

    if new_turn:
        self.agents.append(agent)

        self.token_turns.append(utterance)
        self.entities.append(entities)
        self.lfs.append(lf)
    else:
        self.token_turns[-1].extend(utterance)
        self.entities[-1].extend(entities)
        self.lfs[-1].extend(lf)


def _process_example(self, ex):
    """
    Convert example to turn-based dialogue from each agent's perspective
    Create two Dialogue objects for each example
    """
    dialogue = Dialogue(agent, ex.scenario.kbs[agent],
                    ex.outcome, ex.ex_id, model=self.model)
    for e in ex.events:
        if self.model in ('lf2lf', 'lflm'):
            lf = e.metadata
            proposal = None
            assert lf is not None
            # TODO: hack
            if lf.get('proposal') is not None:
                proposal = {'me': {}, 'you': {}}
                # Parser is from the receiver's perspective
                received_proposal = {int(k): v for k, v in lf['proposal'].iteritems()}
                proposal['me'] = received_proposal[dialogue.agent]
                proposal['you'] = received_proposal[1-dialogue.agent]
            if e.action == 'select':
                if e.agent != dialogue.agent:
                    proposal = None
                else:
                    sel = ex.outcome['item_split']
                    proposal = {'me': {}, 'you': {}}
                    for item, count in sel[dialogue.agent].iteritems():
                        proposal['me'][item] = count
                    for item, count in sel[1-dialogue.agent].iteritems():
                        proposal['you'][item] = count
            #if proposal:
            #    if e.agent == dialogue.agent:
            #        proposal = proposal['me']
            #    else:
            #        proposal = proposal['you']
            utterance = dialogue.lf_to_tokens(lf, proposal, items=self.lexicon.items)
        else:
            sel = ex.outcome['item_split']
            utterance = self.process_event(e, dialogue.agent, sel)
        if utterance:
            dialogue.add_utterance(e.agent, utterance, lf=e.metadata)
    yield dialogue
'''

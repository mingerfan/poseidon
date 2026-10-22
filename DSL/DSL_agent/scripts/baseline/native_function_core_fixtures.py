"""Manual source fixtures for the typed core, never included in provider prompts."""
def declaration(name, arguments, body, signature='c'):
    return '@hc.func('+repr(signature)+')\ndef '+name+'('+arguments+'):\n'+''.join('    '+line+'\n' for line in body.splitlines())


PROGRAMS = {
    'scalar': declaration('helper','x','return x*x+.25')+declaration('golden','x','return helper(x)'),
    'pair': declaration('helper','x','return x+.25,x*.5')+declaration('golden','x','return helper(x)'),
    'nested': declaration('first','x','return x*.5')+declaration('second','x','return first(x)+.25')+
              declaration('golden','x','return second(first(x))'),
    'forward': declaration('golden','x','return helper(x)')+declaration('helper','x','return x*x+.25'),
    'two_inputs': declaration('helper','x,y','return x-y','c,c')+declaration('golden','x,y','return helper(y,x)','c,c'),
    'public_argument': declaration('helper','x,weight','return x*weight','c,p')+declaration('golden','x','return helper(x,.5)'),
    'identity': declaration('helper','x','return x')+declaration('golden','x','return helper(x)'),
    'repeated': declaration('helper','x','return x+.25')+declaration('golden','x','return helper(x)+helper(x)'),
    'zero_input': declaration('helper','','return .5','')+declaration('golden','x','return x*helper()'),
    'empty_helper': declaration('helper','x','return []')+declaration('golden','x','helper(x)\nreturn x+.25'),
    'plain_return': declaration('helper','x','return x','p')+declaration('golden','x','return x*helper(.5)'),
}


def options(name):
    return dict(expected_outputs=2 if name == 'pair' else 1, input_names=('x','y') if name == 'two_inputs' else ('x',))

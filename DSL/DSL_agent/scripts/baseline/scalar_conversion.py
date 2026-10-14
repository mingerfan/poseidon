"""Bounded public scalar extraction. Never convert ciphertexts/Plain Expr to data."""
import math

from seal_artifact_gate import require
from public_numeric import Array, number, LIMIT
import object_arrays as objects


def object_from_numeric(value):
    import numpy as np
    require(type(value) is Array,'Expected public numeric array')
    objects.shape(value.shape)
    return objects.wrap(np.array([number(v) for v in value.values],dtype=object).reshape(value.shape))


def item(value, args):
    require(type(value) in (Array,objects.ObjectArray),'item requires an array, not scalar/Expr')
    require(len(args)<=4,'item index rank limit')
    indices=args[0] if len(args)==1 and type(args[0]) is tuple else args
    require(all(type(i) is int and abs(i)<=LIMIT for i in indices),
            'item indices require bounded public integers, never secret/bool')
    native=object_from_numeric(value) if type(value) is Array else value
    try:
        return objects.scalar(native.data.item(*args))
    except (IndexError,TypeError,ValueError) as error:
        raise ValueError('Invalid item size or index') from error


def cast(name,args):
    require(name in ('float','int'),'Unknown scalar conversion')
    require(len(args)<=(2 if name=='int' else 1),'Invalid scalar conversion arity')
    if not args:return (0.0 if name=='float' else 0),False
    if len(args)==2:
        require(type(args[0]) is str and type(args[1]) in (int,bool),
                'Explicit int base requires public string and integer base')
        require(len(args[0])<=128 and abs(args[1])<=LIMIT,'Numeric conversion resource limit')
        try:return number(int(args[0],args[1])),False
        except (ValueError,TypeError,OverflowError) as error:
            raise ValueError('Invalid integer base/string conversion') from error
    value=args[0]
    deprecated=False
    if type(value) in (Array,objects.ObjectArray):
        dimensions=value.shape
        require(math.prod(dimensions)==1,'Scalar conversion requires exactly one array element')
        value=item(value,[])
        deprecated=bool(dimensions)
    require(type(value) in (bool,int,float,str),'Scalar conversion requires public real/bool/string; no Expr')
    if type(value) is str:require(len(value)<=128,'Numeric string resource limit')
    elif type(value) is not bool:number(value)
    try:
        return number((float if name=='float' else int)(value)),deprecated
    except (ValueError,TypeError,OverflowError) as error:
        raise ValueError('Invalid scalar conversion domain') from error

def array_facts(value):
    if type(value) is Array:return dict(storage='numeric',shape=list(value.shape))
    if type(value) is objects.ObjectArray:return dict(storage='object',shape=list(value.shape))
    return dict(storage=type(value).__name__,shape=None)


def cast_facts(name,args,result,deprecated):
    facts=array_facts(args[0]) if args else dict(storage='default',shape=None)
    source=item(args[0],[]) if args and type(args[0]) in (Array,objects.ObjectArray) else args[0] if args else None
    facts.update(name=name,arity=len(args),deprecated=deprecated,
                 negative_fraction=type(source) is float and source<0 and source!=int(source),
                 result_type=type(result).__name__)
    return facts


def item_facts(receiver,args,result):
    facts=array_facts(receiver)
    facts.update(index_form='sole' if not args else 'tuple' if len(args)==1 and type(args[0]) is tuple else
                 'flat' if len(args)==1 else 'positional',
                 negative_index=any(v<0 for v in (args[0] if len(args)==1 and type(args[0]) is tuple else args)),
                 result_kind=getattr(result,'kind',type(result).__name__))
    return facts


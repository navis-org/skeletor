#    This script is part of skeletor (http://www.github.com/navis-org/skeletor).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.

"""Optional acceleration through ``navis-fastcore``.

``navis-fastcore`` ships Rust implementations of the graph primitives skeletor
would otherwise get from ``igraph``/``networkx``/``scipy``: connected
components, level-set components, vertex contraction, minimum spanning trees
and geodesic distances, all operating straight off an edge list. Where a
primitive is available we use it; where it isn't we fall back to the original
implementation.

The two are equivalent but not always bit-identical, for two reasons:

1. Summing the same values in a different order can move a result by an ulp.
   Where that only affects a leaf value (a radius, say) it is harmless; where it
   would propagate we keep the two paths exactly in step on purpose - see
   :func:`skeletor.utilities._edge_lengths` for one such case.
2. Some of these problems have more than one correct answer. Minimum spanning
   trees over tied edge weights are the big one: the two implementations break
   those ties differently, so `by_wavefront` with several waves can return trees
   that differ in a few percent of their edges - at identical total weight, so
   both are valid. `by_tangent_ball` likewise picks a different, equally
   arbitrary target among equidistant ones, and `by_teasar` maps a vertex that
   sits exactly between two nodes of the path invalidating it to either one
   (~0.2% of vertices on the example mesh). Individual call sites note these.

The primitives were added at different times, so testing that the module
imports is not enough - each call site must check for the specific feature it
wants via :func:`has`::

    from .. import _fastcore

    if _fastcore.has('level_set_components'):
        ids, n = _fastcore.fastcore.level_set_components(edges, n_nodes, labels)
    else:
        ...  # igraph fallback

Some of those features are methods rather than functions - ``GeodesicGraph`` is
the same primitives with the per-call adjacency build hoisted out, for call sites
that ask many small questions of one graph, and its methods likewise arrived at
different times. :func:`has` takes a dotted name for those::

    if _fastcore.has('GeodesicGraph.set_weights'):
        ...

Arguments arrive late too, and a function that exists but does not take the
keyword you need is no use. Name it in parentheses to ask for it::

    if _fastcore.has('geodesic_matrix_graph(dtype)'):
        ...

"""

import functools
import inspect


@functools.lru_cache(maxsize=None)
def _parameters(obj):
    """Parameter names of ``obj``, or an empty set if it has no signature.

    Cached because :func:`has` is called per connected component at some call
    sites and ``inspect.signature`` on a compiled function costs ~9 us against
    ~0.4 us for the attribute lookup around it. Keyed on the resolved object
    rather than the feature name, so swapping the ``fastcore`` module out - which
    the tests do to exercise the fallbacks - is not cached over.
    """
    try:
        return frozenset(inspect.signature(obj).parameters)
    except (TypeError, ValueError):
        return frozenset()

try:
    import navis_fastcore as fastcore
except ImportError:
    fastcore = None
except BaseException:
    raise

__all__ = ['fastcore', 'has']


def has(*funcs):
    """Check whether fastcore is installed and provides all of ``funcs``.

    Parameters
    ----------
    *funcs :    str
                Names of the ``navis_fastcore`` features required by the caller.
                A dotted name walks into the attribute, so
                ``'GeodesicGraph.set_weights'`` asks for the method rather than
                just the class - which a call site needing it has to do, since
                the class grew its methods at different times. A trailing
                ``(arg)`` additionally requires that keyword argument, for the
                call sites that need one that arrived after the function did.

    Returns
    -------
    bool

    """
    def resolve(spec):
        # Parse before resolving: a malformed spec is a typo in skeletor, and
        # silently reporting the feature missing would pin the call site to its
        # fallback forever with nothing to notice it by.
        name, _, args = spec.partition('(')
        wanted = ()
        if args:
            if not args.endswith(')'):
                raise ValueError(f'Malformed feature spec "{spec}" - expected '
                                 '`func(arg)` or `func(arg, arg)`')
            wanted = tuple(a.strip() for a in args[:-1].split(','))
            if not all(wanted):
                raise ValueError(f'Empty argument name in feature spec "{spec}"')

        obj = fastcore
        for part in name.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                return False

        return not wanted or set(wanted) <= _parameters(obj)

    if fastcore is None:
        return False
    return all(resolve(f) for f in funcs)

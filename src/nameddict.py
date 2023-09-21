
class nameddict(dict):
   """
   Examples
   --------
   >>> nameddict({'a':1, 'b':2})
   {'a': 1, 'b': 2}
   >>> x = nameddict(a=1, b=2)
   >>> x.a
   1
   >>> x['a']
   1
   >>> x.translate(3)
   ['a', 'b']

   """
   __getattr__ = dict.__getitem__

   def translate(self, x):
      return [name for name,f in self.items() if (f & x) or f==x==0]

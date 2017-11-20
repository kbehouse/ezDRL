import yaml

cfg = []

""" For loader tuple"""
class YAMLPatch(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

YAMLPatch.add_constructor(u'tag:yaml.org,2002:python/tuple', YAMLPatch.construct_python_tuple)


""" Load File"""
with open("config/default_config.yaml", 'r') as stream:
    try:
        cfg = yaml.load(stream, Loader=YAMLPatch)
    except yaml.YAMLError as exc:
        print(exc)

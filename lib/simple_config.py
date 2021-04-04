from copy import deepcopy
from typing import List, Tuple, Dict
from dataclasses import dataclass
import yaml


class SimpleConfig:
    def __init__(self):
        super(SimpleConfig, self).__init__()

    def check(self):
        self.check_type()
        self.check_value()

    def check_value(self):
        pass

    def check_type(self, obj=None):
        basic_types = [bool, int, float, str]
        if obj is None: obj = self
        assert issubclass(type(obj), SimpleConfig), f'this method checks types of {type(self)} obj instead of {type(obj)} object'
        assert hasattr(obj, '__dict__'), f'object does not have any local attribute'
        assert hasattr(obj, '__annotations__'), f'object of class {type(obj)} has no type annotation'
        for var_name, var in obj.__dict__.items():
            var_anno_type = obj.__annotations__[var_name]
            if type(var) in basic_types:
                assert type(var) is var_anno_type, \
                    f'actual type {type(var)} of attribute {var_name} is inconsisent with its annotation type {var_anno_type}'
            elif type(var) in [list, tuple]:
                assert type(var[0]) in basic_types, f'unecpected type {type(var[0])} of items in list/tuple {var_name}'
                assert all([type(i) == type(var[0]) for i in var]), f'items in list/tuple {var_name} are not exactly the same'
                assert List[type(var[0])] == var_anno_type or Tuple[type(var[0])] == var_anno_type, \
                    f'actual type {type(var)} of attribute {var_name} is inconsisent with its annotation type {var_anno_type}'
            elif issubclass(type(var), SimpleConfig):
                obj.check_type(var)
            else:
                raise AssertionError(f'unexpected type {type(var)} of attirbute {var_name}')

    def merge_with_dotdict(self, dotdict: Dict):
        for keys_seq, var in dotdict.items():
            assert type(keys_seq) == str, f'unexpected arg format: "{keys_seq: var}"'

            keys = keys_seq.split('.')
            sub_obj = self
            for k_idx, k in enumerate(keys):
                try:
                    assert k in sub_obj.__dict__
                except Exception as e:
                    raise AssertionError(f'attribute "{k}" does not exists')
                if k_idx != len(keys) - 1:
                    sub_obj = getattr(sub_obj, k)

            setattr(sub_obj, keys[-1], deepcopy(var))

        self.check()
        return True

    def merge_with_dotlist(self, dotlist: List[str]):
        dotdict = {}
        for arg in dotlist:
            try:
                keys_seq, var = arg.split('=')
            except Exception as e:
                raise ValueError(f'unexpected arg format: "{arg}"')

            if keys_seq.startswith('--'): keys_seq = keys_seq[2:]

            if var[0] == '[' and var[-1] == ']':
                dotdict[keys_seq] = [self.format_str(i) for i in var[1:-1].split(',')]
            elif var[0] == '(' and var[-1] == ')':
                dotdict[keys_seq] = (self.format_str(i) for i in var[1:-1].split(','))
            else:
                dotdict[keys_seq] = self.format_str(var)

        return self.merge_with_dotdict(dotdict)

    def merge_with_dict(self, dict_obj: Dict):
        dotdict = self.dict_to_dotdict(dict_obj)
        return self.merge_with_dotdict(dotdict)

    def merge_with_yaml(self, yaml_path):
        yaml_dict = yaml.safe_load(open(yaml_path))
        return self.merge_with_dict(yaml_dict)

    def to_dict(self, obj=None):
        if obj is None: obj = self
        else: assert isinstance(obj, SimpleConfig)
        dict_obj = {}
        for key, var in obj.__dict__.items():
            if issubclass(type(var), SimpleConfig):
                dict_obj[key] = obj.to_dict(var)
            else:
                dict_obj[key] = var
        return dict_obj

    def to_yaml(self):
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def dict_to_dotdict(cls, dict_obj: Dict):
        dotdict = {}
        for key, var in dict_obj.items():
            assert isinstance(key, str), f'unexpected type {type(key)} of key {key} of dict'
            if isinstance(var, dict):
                sub_dotdict = cls.dict_to_dotdict(var)
                dotdict.update({key + '.' + k: v for k, v in sub_dotdict.items()})
            else:
                dotdict[key] = var
        return dotdict

    @staticmethod
    def format_str(string: str):
        string = string.strip()
        if (string[0] == '"' and string[-1] == '"') or (string[0] == "'" and string[-1] == "'"):
            string = string[1:-1]
        elif string == 'True' or string == 'true': return True
        elif string == 'False' or string == 'false': return False
        else:
            try:
                string = int(string)
            except ValueError:
                try:
                    string = float(string)
                except ValueError:
                    pass
        return string


def main_t():
    @dataclass
    class SubConfig(SimpleConfig):
        t: float = 2.0
        t2: List[float] = (2.0, 3.0)

    @dataclass
    class Config(SimpleConfig):
        x: int = 3
        model_name: SubConfig = SubConfig()

        def __post_init__(self):
            self.check()

    cfg = Config()
    dotdict = {'model_name.t2': [2.0, 3.0], 'x': 4}
    dotlist = ['model_name.t2=[2.0, 3.0]', 'x=4']
    cfg.merge_with_dotlist(dotlist)
    dict_obj = cfg.to_dict()
    print(cfg)


def dict_to_dotdict_t():
    f = {'A': {'a': {'1': [2, 3]}}, 'B': {'a': 't'}, 'C': {'a': 1, 'b': {'1': 2, '2': [3,4]}}}
    x = SimpleConfig.dict_to_dotdict(f)
    print('Done')


if __name__ == '__main__':
    main_t()

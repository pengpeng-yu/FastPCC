from copy import deepcopy
from typing import List, Tuple, Dict
from dataclasses import dataclass
import yaml


class SimpleConfig:
    def __init__(self):
        super(SimpleConfig, self).__init__()

    def check(self):
        """
        Called by a series of merge_xxx func. You can also call it in __init__().
        """
        self.check_type()
        self.check_value()

    def check_value(self):
        """
        Recursively call check_local_value() of all attributes to check value. Only callled by check().
        """
        self.check_local_value()
        for key, value in self.__dict__.items():
            if issubclass(type(value), SimpleConfig):
                value.check_value()

    def check_local_value(self):
        """
        This func could be overrided in subclass to check value of attributes. Only callled by check_value().
        """
        pass

    def check_type(self):
        """
        Recursively check existence and value type of attributes. Only called by check().
        """
        basic_types = [bool, int, float, str]
        assert issubclass(type(self), SimpleConfig), f'this method checks types of {type(self)} obj instead of {type(self)} object'
        assert hasattr(self, '__dict__'), f'object does not have any local attribute'
        assert hasattr(self, '__annotations__'), f'object of class {type(self)} has no type annotation'
        for key, value in self.__dict__.items():
            value_anno_type = self.__annotations__[key]
            if type(value) in basic_types:
                assert type(value) is value_anno_type, \
                    f'actual type {type(value)} of attribute {key} is inconsisent with its annotation type {value_anno_type}'
            elif type(value) in [list, tuple]:
                assert type(value[0]) in basic_types, f'unecpected type {type(value[0])} of items in list/tuple {key}'
                assert all([type(i) == type(value[0]) for i in value]), f'items in list/tuple {key} are not exactly the same'
                assert List[type(value[0])] == value_anno_type or Tuple[type(value[0])] == value_anno_type, \
                    f'actual type {type(value)} of attribute {key} is inconsisent with its annotation type {value_anno_type}'
            elif issubclass(type(value), SimpleConfig):
                value.check_type()
            else:
                raise AssertionError(f'unexpected type {type(value)} of attirbute {key}')

    def merge_setattr(self, key, value):
        self.__dict__[key] = value

    def merge_with_dotdict(self, dotdict: Dict):
        """
        dotdict: {'a': 2, 'b.a': 'string', '-b.b.a_a': [1,2,3], '--b.b.b-b': ['1', '2']}
        Value will not be formatted.
        """
        for keys_seq, value in dotdict.items():
            assert type(keys_seq) == str, f'unexpected arg format: "{keys_seq}: {value}"'
            if keys_seq.startswith('--'): keys_seq = keys_seq[2:]
            keys_seq.replace('-', '_')

            if '.' not in keys_seq:
                try:
                    assert keys_seq in self.__dict__
                except Exception as e:
                    raise AssertionError(f'attribute "{keys_seq}" does not exists')
                self.merge_setattr(keys_seq, deepcopy(value))
            else:
                first_key, res_keys = keys_seq.split('.', 1)
                try:
                    assert issubclass(type(self.__dict__[first_key]), SimpleConfig)
                except Exception as e:
                    raise AssertionError(f'attribute "{first_key}" is not a subclass of SimpleConfig')
                self.__dict__[first_key].merge_with_dotdict({res_keys: value})

        self.check()
        return True

    def merge_with_dotlist(self, dotlist: List[str]):
        """
        dotlist: ['a==2', 'b.a=string', '-b.b.a_a=[1,2,3]', '--b.b.b-b==["1","2"]']
        Value will be formatted.
        """
        dotdict = {}
        for arg in dotlist:
            arg = arg.replace('==', '=')
            try:
                keys_seq, var = arg.split('=')
            except Exception as e:
                raise ValueError(f'unexpected arg format: "{arg}"')

            if var[0] == '[' and var[-1] == ']':
                dotdict[keys_seq] = [self.format_str(i) for i in var[1:-1].split(',')]
            elif var[0] == '(' and var[-1] == ')':
                dotdict[keys_seq] = (self.format_str(i) for i in var[1:-1].split(','))
            else:
                dotdict[keys_seq] = self.format_str(var)

        return self.merge_with_dotdict(dotdict)

    def merge_with_dict(self, dict_obj: Dict):
        """
        dict:  {
                'a': 2,
                'b': {
                    'a': "string",
                    'b': {
                            'a_a': [1,2,3],
                            'b_b': ["1", "2"]
                        }
                    }
                }
        Value will not be formatted.
        """
        dotdict = self.dict_to_dotdict(dict_obj)
        return self.merge_with_dotdict(dotdict)

    def merge_with_yaml(self, yaml_path):
        yaml_dict = yaml.safe_load(open(yaml_path))
        return self.merge_with_dict(yaml_dict)

    def to_dict(self):
        assert isinstance(self, SimpleConfig)
        dict_obj = {}
        for key, var in self.__dict__.items():
            if issubclass(type(var), SimpleConfig):
                dict_obj[key] = var.to_dict()
            else:
                dict_obj[key] = var
        return dict_obj

    def to_yaml(self):
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def dict_to_dotdict(self, dict_obj: Dict):
        dotdict = {}
        for key, var in dict_obj.items():
            assert isinstance(key, str), f'unexpected type {type(key)} of key {key} of dict'
            if isinstance(var, dict):
                sub_dotdict = self.dict_to_dotdict(var)
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


def dict_to_dotdict_t():
    f = {'A': {'a': {'1': [2, 3]}}, 'B': {'a': 't'}, 'C': {'a': 1, 'b': {'1': 2, '2': [3,4]}}}
    x = SimpleConfig.dict_to_dotdict(f)
    print('Done')


if __name__ == '__main__':
    dict_to_dotdict_t()

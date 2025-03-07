from copy import deepcopy
from typing import List, Tuple, Dict, Union
import importlib
import yaml


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class SimpleConfig:
    def __init__(self):
        super(SimpleConfig, self).__init__()
        self.local_auto_import()

    def check(self):
        """
        You are supposed to call this function after building a config object to check attributes.
        """
        self.check_type()
        self.check_value()

    def check_value(self):
        """
        Recursively call check_local_value() of all attributes to check value. Only called by check().
        """
        self.check_local_value()
        for key, value in self.__dict__.items():
            if issubclass(type(value), SimpleConfig):
                value.check_value()

    def check_local_value(self):
        """
        This func could be overridden in subclass to check value of attributes. Only called by check_value().
        """
        pass

    def check_type(self):
        """
        Recursively check existence and value type of attributes. Only called by check().

        Only
        basic_types (bool, int, float, str),
        Unions of basic_types,
        List[basic_type], Tuple[basic_type, ...],
        Union[basic_type, List[basic_type]], Union[basic_type, Tuple[basic_type, ...]],
        are supported.

        List[basic_type] and Tuple[basic_type, ...] are both
        interpreted as Union[List[basic_type], Tuple[basic_type, ...]]
        """
        basic_types = [bool, int, float, str]
        assert issubclass(type(self), SimpleConfig), \
            f'this method checks types of {type(self)} obj instead of {type(self)} object'
        assert hasattr(self, '__dict__'), f'object does not have attribute __dict__'
        if self.__dict__ != {} and not hasattr(self, '__dataclass_fields__'):
            assert hasattr(self, '__annotations__'), f'object of class {type(self)} has no type annotation'

        for key, value in self.__dict__.items():
            value_anno_type = self.get_anno_by_key(key)
            value_type = type(value)

            if value_type in basic_types:
                is_valid_type = False
                if value_type == value_anno_type:
                    is_valid_type = True
                elif hasattr(value_anno_type, '__origin__') and value_anno_type.__origin__ == Union:
                    for sub_type in value_anno_type.__args__:
                        if value_type == sub_type:
                            is_valid_type = True
                            break
                if not is_valid_type:
                    raise TypeError(
                        f'actual type {value_type} of attribute {key} is '
                        f'inconsistent with its annotation type {value_anno_type}'
                    )

            elif value_type in [list, tuple]:
                element_type = value_anno_type.__args__[0]
                assert element_type in basic_types, \
                    f'unexpected type {element_type} of items in list/tuple {key}'
                assert all([type(i) == element_type for i in value]), \
                    f'items in list/tuple {key} are not exactly the same'
                assert value_anno_type in (
                    List[element_type],
                    Tuple[element_type, ...],
                    Union[element_type, List[element_type]],
                    Union[element_type, Tuple[element_type, ...]]
                ), f'actual type {value_type} of attribute {key} ' \
                   f'is inconsistent with its annotation type {value_anno_type}'

            elif issubclass(value_type, SimpleConfig):
                assert issubclass(value_anno_type, SimpleConfig)
                value.check_type()

            else:
                raise AssertionError(f'unexpected type {value_type} of attribute {key}')

    def local_auto_import(self, keys=None):
        """
        Automatically called by merge_setattr()
        to import config object for keys with a corresponding path_key to be merged.
        You can manually call this function to import SimpleConfig class
        without using merge_setattr().
        """
        if keys is None: keys = self.__dict__
        elif isinstance(keys, str):
            keys = [keys]

        for key in keys:
            if isinstance(key, str) and key.endswith('_module_path'):
                target_key = key[:-len('_module_path')]
                if target_key in self.__dict__:
                    assert issubclass(self.get_anno_by_key(target_key), SimpleConfig)
                    try:
                        self.__dict__[target_key] = importlib.import_module(self.__dict__[key]).Config()
                        assert issubclass(type(self.__dict__[target_key]), SimpleConfig)
                    except Exception as e:
                        raise ImportError(*e.args)

    def get_anno_by_key(self, key):
        if hasattr(self, '__dataclass_fields__'):
            return self.__dataclass_fields__[key].type
        else:
            return self.__annotations__[key]

    def merge_setattr(self, key, value):
        """
        Modifications of values are supposed to be implemented using this function.
        """
        self.__dict__[key] = value
        self.local_auto_import(key)

    def merge_with_dotdict(self, dotdict: Dict):
        """
        Fundamental function for merging configs.
        A dotdict: {'a': 2, 'b.a': 'string', '-b.b.a_a': [1,2,3], '--b.b.b-b': ['1', '2']}
        Values will not be formatted.
        """
        for keys_seq, value in dotdict.items():
            assert type(keys_seq) == str, f'unexpected arg format: "{keys_seq}: {value}"'
            if keys_seq.startswith('--'): keys_seq = keys_seq[2:]
            keys_seq = keys_seq.replace('-', '_')

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

        return True

    def merge_with_dict(self, dict_obj: Dict):
        """
        A dict:
        {
            'a': 2,
            'b': {
                'a': "string",
                'b': {
                    'a_a': [1,2,3],
                    'b-b': ["1", "2"]
                }
            }
        }
        Values will not be formatted.
        """
        dotdict = self.dict_to_dotdict(dict_obj)
        return self.merge_with_dotdict(dotdict)

    def merge_with_yaml(self, yaml_path):
        """
        # A yaml string:
        # include "path/to/config/base_config_1.yaml"
        # include "path/to/config/base_config_2"
        a: 2
        b:
            a: "string",
            b:
                a_a: [1, 2, 3]
        b.b.b-b': ['1', '2']
        # Values are formatted by pyyaml.
        """
        try:
            f = open(yaml_path)
        except FileNotFoundError as e:
            if not yaml_path.endswith('.yaml'):
                f = open(yaml_path + '.yaml')
            else:
                raise e
        f_left = ''
        for line in f:
            if line.startswith('# include') or line.startswith('#include'):
                sub_yaml_path = self.format_str(line.rstrip().split(' ', 2)[-1])
                self.merge_with_yaml(sub_yaml_path)
            # elif line.strip() == '' or line.lstrip()[0] == '#':
            #     pass
            else:
                f_left = line + f.read()
                break
        f.close()
        return self.merge_with_dotdict(self.yaml_str_to_dotdict(f_left))

    def merge_with_dotdict_list(self, dotdict_list: List[Dict]):
        """
        dotdict_list: sequence of dotdict to be merged in order.
        Values will not be formatted.
        """
        ret = True
        for dotdict in dotdict_list:
            ret &= self.merge_with_dotdict(dotdict)
        return ret

    def merge_with_dotlist(self, dotlist: List[str]):
        """
        dotlist: ['a==2', 'b.a=string', '-b.b.a_a=[1,2,3]', '--b.b.b-b==["1","2"]']
                sequence of key-value pairs to be merged in order.
        YAML file paths are supported.
        Repeated keys are allowed by treating each element in a dotlist as a dotdict.
        Values will be formatted.
        """
        dotdict_list = []
        for arg in dotlist:
            arg = arg.replace('==', '=')
            try:
                keys_seq, var = arg.split('=')
            except ValueError:
                try:
                    self.merge_with_yaml(arg)
                except Exception as e:
                    raise ValueError(f'unexpected arg: "{arg}"')
            else:
                if var == '':
                    dotdict_list.append({keys_seq: var})
                elif var[0] == '[' and var[-1] == ']':
                    var = var[1:-1]
                    if var[-1] == ',':
                        var = var[:-1]
                    dotdict_list.append(
                        {keys_seq: [self.format_str(i) for i in var.split(',')]}
                    )
                elif var[0] == '(' and var[-1] == ')':
                    var = var[1:-1]
                    if var[-1] == ',':
                        var = var[:-1]
                    dotdict_list.append(
                        {keys_seq: tuple(self.format_str(i) for i in var.split(','))}
                    )
                else:
                    dotdict_list.append(
                        {keys_seq: self.format_str(var)}
                    )

        return self.merge_with_dotdict_list(dotdict_list)

    def to_dict(self):
        dict_obj = {}
        for key, var in self.__dict__.items():
            if issubclass(type(var), SimpleConfig):
                dict_obj[key] = var.to_dict()
            else:
                dict_obj[key] = var
        return dict_obj

    def to_yaml(self):
        return yaml.dump(
            self.to_dict(),
            Dumper=NoAliasDumper,
            default_flow_style=False,
            sort_keys=False
        )

    def yaml_str_to_dotdict(self, yaml_str):
        yaml_dict = yaml.safe_load(yaml_str) or {}
        dotdict = self.dict_to_dotdict(yaml_dict)
        return dotdict

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
        if string == '':
            pass
        elif (string[0] == '"' and string[-1] == '"') or (string[0] == "'" and string[-1] == "'"):
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
    f = {'A': {'a': {'1': [2, 3]}}, 'B': {'a': 't'}, 'C': {'a': 1, 'b': {'1': 2, '2': [3, 4]}}}
    x = SimpleConfig.dict_to_dotdict(f)
    print('Done')


if __name__ == '__main__':
    dict_to_dotdict_t()

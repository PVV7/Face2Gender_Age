from typing import List, Tuple, Dict, Any

age_dict = {
    0: '0-2',
    1: '3-5',
    2: '6-8',
    3: '9-11',
    4: '12-14',
    5: '15-19',
    6: '20-24',
    7: '25-29',
    8: '30-34',
    9: '35-39',
    10: '40-44',
    11: '45-49',
    12: '50-54',
    13: '55-59',
    14: '60-64',
    15: '65-69',
    16: 'more than 70'}

gender_dict = {
    0: 'Пол: Мужской',
    1: 'Пол: Женский'
}


def convert_data(lst_data: List, to_str: bool=False) -> Any:
    lst_new_data = []

    for age, gender in lst_data:
        lst_new_data.append((age_dict[age], gender_dict[gender]))

    if to_str:
        data_string = '\n'.join(map(lambda x: ' года, '.join(x), lst_new_data))
        return data_string

    return lst_new_data
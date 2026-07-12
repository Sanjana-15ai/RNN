"""
Country Information Generator — One-to-Many RNN Demo
Single-file Streamlit application.

Concept: A One-to-Many Recurrent Neural Network takes ONE input (a country
name, encoded as a fixed-length context vector) and unrolls across several
timesteps, emitting a different piece of information at each step
(Capital -> Currency -> Population -> Language -> Famous Places).

This app demonstrates that architecture end-to-end:
 1. The chosen country name is embedded (simulated) into a context vector.
 2. A hidden state h0 is initialized from that context.
 3. The RNN is "unrolled" for 5 timesteps, each one producing one field of
    the country profile, visualized as an animated diagram.
 4. The generated sequence is rendered as a set of premium profile cards.

All data (population, region, capital, currency, language, landmarks) is
embedded directly in this file — no external files or network calls needed.
"""

import time
import random
import streamlit as st

POPULATION = {
    'Afghanistan': 31056997,
    'Albania': 3581655,
    'Algeria': 32930091,
    'American Samoa': 57794,
    'Andorra': 71201,
    'Angola': 12127071,
    'Anguilla': 13477,
    'Antigua & Barbuda': 69108,
    'Argentina': 39921833,
    'Armenia': 2976372,
    'Aruba': 71891,
    'Australia': 20264082,
    'Austria': 8192880,
    'Azerbaijan': 7961619,
    'Bahamas, The': 303770,
    'Bahrain': 698585,
    'Bangladesh': 147365352,
    'Barbados': 279912,
    'Belarus': 10293011,
    'Belgium': 10379067,
    'Belize': 287730,
    'Benin': 7862944,
    'Bermuda': 65773,
    'Bhutan': 2279723,
    'Bolivia': 8989046,
    'Bosnia & Herzegovina': 4498976,
    'Botswana': 1639833,
    'Brazil': 188078227,
    'British Virgin Is.': 23098,
    'Brunei': 379444,
    'Bulgaria': 7385367,
    'Burkina Faso': 13902972,
    'Burma': 47382633,
    'Burundi': 8090068,
    'Cambodia': 13881427,
    'Cameroon': 17340702,
    'Canada': 33098932,
    'Cape Verde': 420979,
    'Cayman Islands': 45436,
    'Central African Rep.': 4303356,
    'Chad': 9944201,
    'Chile': 16134219,
    'China': 1313973713,
    'Colombia': 43593035,
    'Comoros': 690948,
    'Congo, Dem. Rep.': 62660551,
    'Congo, Repub. of the': 3702314,
    'Cook Islands': 21388,
    'Costa Rica': 4075261,
    "Cote d'Ivoire": 17654843,
    'Croatia': 4494749,
    'Cuba': 11382820,
    'Cyprus': 784301,
    'Czech Republic': 10235455,
    'Denmark': 5450661,
    'Djibouti': 486530,
    'Dominica': 68910,
    'Dominican Republic': 9183984,
    'East Timor': 1062777,
    'Ecuador': 13547510,
    'Egypt': 78887007,
    'El Salvador': 6822378,
    'Equatorial Guinea': 540109,
    'Eritrea': 4786994,
    'Estonia': 1324333,
    'Ethiopia': 74777981,
    'Faroe Islands': 47246,
    'Fiji': 905949,
    'Finland': 5231372,
    'France': 60876136,
    'French Guiana': 199509,
    'French Polynesia': 274578,
    'Gabon': 1424906,
    'Gambia, The': 1641564,
    'Gaza Strip': 1428757,
    'Georgia': 4661473,
    'Germany': 82422299,
    'Ghana': 22409572,
    'Gibraltar': 27928,
    'Greece': 10688058,
    'Greenland': 56361,
    'Grenada': 89703,
    'Guadeloupe': 452776,
    'Guam': 171019,
    'Guatemala': 12293545,
    'Guernsey': 65409,
    'Guinea': 9690222,
    'Guinea-Bissau': 1442029,
    'Guyana': 767245,
    'Haiti': 8308504,
    'Honduras': 7326496,
    'Hong Kong': 6940432,
    'Hungary': 9981334,
    'Iceland': 299388,
    'India': 1095351995,
    'Indonesia': 245452739,
    'Iran': 68688433,
    'Iraq': 26783383,
    'Ireland': 4062235,
    'Isle of Man': 75441,
    'Israel': 6352117,
    'Italy': 58133509,
    'Jamaica': 2758124,
    'Japan': 127463611,
    'Jersey': 91084,
    'Jordan': 5906760,
    'Kazakhstan': 15233244,
    'Kenya': 34707817,
    'Kiribati': 105432,
    'Korea, North': 23113019,
    'Korea, South': 48846823,
    'Kuwait': 2418393,
    'Kyrgyzstan': 5213898,
    'Laos': 6368481,
    'Latvia': 2274735,
    'Lebanon': 3874050,
    'Lesotho': 2022331,
    'Liberia': 3042004,
    'Libya': 5900754,
    'Liechtenstein': 33987,
    'Lithuania': 3585906,
    'Luxembourg': 474413,
    'Macau': 453125,
    'Macedonia': 2050554,
    'Madagascar': 18595469,
    'Malawi': 13013926,
    'Malaysia': 24385858,
    'Maldives': 359008,
    'Mali': 11716829,
    'Malta': 400214,
    'Marshall Islands': 60422,
    'Martinique': 436131,
    'Mauritania': 3177388,
    'Mauritius': 1240827,
    'Mayotte': 201234,
    'Mexico': 107449525,
    'Micronesia, Fed. St.': 108004,
    'Moldova': 4466706,
    'Monaco': 32543,
    'Mongolia': 2832224,
    'Montserrat': 9439,
    'Morocco': 33241259,
    'Mozambique': 19686505,
    'Namibia': 2044147,
    'Nauru': 13287,
    'Nepal': 28287147,
    'Netherlands': 16491461,
    'Netherlands Antilles': 221736,
    'New Caledonia': 219246,
    'New Zealand': 4076140,
    'Nicaragua': 5570129,
    'Niger': 12525094,
    'Nigeria': 131859731,
    'N. Mariana Islands': 82459,
    'Norway': 4610820,
    'Oman': 3102229,
    'Pakistan': 165803560,
    'Palau': 20579,
    'Panama': 3191319,
    'Papua New Guinea': 5670544,
    'Paraguay': 6506464,
    'Peru': 28302603,
    'Philippines': 89468677,
    'Poland': 38536869,
    'Portugal': 10605870,
    'Puerto Rico': 3927188,
    'Qatar': 885359,
    'Reunion': 787584,
    'Romania': 22303552,
    'Russia': 142893540,
    'Rwanda': 8648248,
    'Saint Helena': 7502,
    'Saint Kitts & Nevis': 39129,
    'Saint Lucia': 168458,
    'St Pierre & Miquelon': 7026,
    'Saint Vincent and the Grenadines': 117848,
    'Samoa': 176908,
    'San Marino': 29251,
    'Sao Tome & Principe': 193413,
    'Saudi Arabia': 27019731,
    'Senegal': 11987121,
    'Serbia': 9396411,
    'Seychelles': 81541,
    'Sierra Leone': 6005250,
    'Singapore': 4492150,
    'Slovakia': 5439448,
    'Slovenia': 2010347,
    'Solomon Islands': 552438,
    'Somalia': 8863338,
    'South Africa': 44187637,
    'Spain': 40397842,
    'Sri Lanka': 20222240,
    'Sudan': 41236378,
    'Suriname': 439117,
    'Swaziland': 1136334,
    'Sweden': 9016596,
    'Switzerland': 7523934,
    'Syria': 18881361,
    'Taiwan': 23036087,
    'Tajikistan': 7320815,
    'Tanzania': 37445392,
    'Thailand': 64631595,
    'Togo': 5548702,
    'Tonga': 114689,
    'Trinidad & Tobago': 1065842,
    'Tunisia': 10175014,
    'Turkey': 70413958,
    'Turkmenistan': 5042920,
    'Turks & Caicos Is': 21152,
    'Tuvalu': 11810,
    'Uganda': 28195754,
    'Ukraine': 46710816,
    'United Arab Emirates': 2602713,
    'United Kingdom': 60609153,
    'United States': 298444215,
    'Uruguay': 3431932,
    'Uzbekistan': 27307134,
    'Vanuatu': 208869,
    'Venezuela': 25730435,
    'Vietnam': 84402966,
    'Virgin Islands': 108605,
    'Wallis and Futuna': 16025,
    'West Bank': 2460492,
    'Western Sahara': 273008,
    'Yemen': 21456188,
    'Zambia': 11502010,
    'Zimbabwe': 12236805,
}

REGION = {
    'Afghanistan': 'ASIA (EX. NEAR EAST)',
    'Albania': 'EASTERN EUROPE',
    'Algeria': 'NORTHERN AFRICA',
    'American Samoa': 'OCEANIA',
    'Andorra': 'WESTERN EUROPE',
    'Angola': 'SUB-SAHARAN AFRICA',
    'Anguilla': 'LATIN AMER. & CARIB',
    'Antigua & Barbuda': 'LATIN AMER. & CARIB',
    'Argentina': 'LATIN AMER. & CARIB',
    'Armenia': 'C.W. OF IND. STATES',
    'Aruba': 'LATIN AMER. & CARIB',
    'Australia': 'OCEANIA',
    'Austria': 'WESTERN EUROPE',
    'Azerbaijan': 'C.W. OF IND. STATES',
    'Bahamas, The': 'LATIN AMER. & CARIB',
    'Bahrain': 'NEAR EAST',
    'Bangladesh': 'ASIA (EX. NEAR EAST)',
    'Barbados': 'LATIN AMER. & CARIB',
    'Belarus': 'C.W. OF IND. STATES',
    'Belgium': 'WESTERN EUROPE',
    'Belize': 'LATIN AMER. & CARIB',
    'Benin': 'SUB-SAHARAN AFRICA',
    'Bermuda': 'NORTHERN AMERICA',
    'Bhutan': 'ASIA (EX. NEAR EAST)',
    'Bolivia': 'LATIN AMER. & CARIB',
    'Bosnia & Herzegovina': 'EASTERN EUROPE',
    'Botswana': 'SUB-SAHARAN AFRICA',
    'Brazil': 'LATIN AMER. & CARIB',
    'British Virgin Is.': 'LATIN AMER. & CARIB',
    'Brunei': 'ASIA (EX. NEAR EAST)',
    'Bulgaria': 'EASTERN EUROPE',
    'Burkina Faso': 'SUB-SAHARAN AFRICA',
    'Burma': 'ASIA (EX. NEAR EAST)',
    'Burundi': 'SUB-SAHARAN AFRICA',
    'Cambodia': 'ASIA (EX. NEAR EAST)',
    'Cameroon': 'SUB-SAHARAN AFRICA',
    'Canada': 'NORTHERN AMERICA',
    'Cape Verde': 'SUB-SAHARAN AFRICA',
    'Cayman Islands': 'LATIN AMER. & CARIB',
    'Central African Rep.': 'SUB-SAHARAN AFRICA',
    'Chad': 'SUB-SAHARAN AFRICA',
    'Chile': 'LATIN AMER. & CARIB',
    'China': 'ASIA (EX. NEAR EAST)',
    'Colombia': 'LATIN AMER. & CARIB',
    'Comoros': 'SUB-SAHARAN AFRICA',
    'Congo, Dem. Rep.': 'SUB-SAHARAN AFRICA',
    'Congo, Repub. of the': 'SUB-SAHARAN AFRICA',
    'Cook Islands': 'OCEANIA',
    'Costa Rica': 'LATIN AMER. & CARIB',
    "Cote d'Ivoire": 'SUB-SAHARAN AFRICA',
    'Croatia': 'EASTERN EUROPE',
    'Cuba': 'LATIN AMER. & CARIB',
    'Cyprus': 'NEAR EAST',
    'Czech Republic': 'EASTERN EUROPE',
    'Denmark': 'WESTERN EUROPE',
    'Djibouti': 'SUB-SAHARAN AFRICA',
    'Dominica': 'LATIN AMER. & CARIB',
    'Dominican Republic': 'LATIN AMER. & CARIB',
    'East Timor': 'ASIA (EX. NEAR EAST)',
    'Ecuador': 'LATIN AMER. & CARIB',
    'Egypt': 'NORTHERN AFRICA',
    'El Salvador': 'LATIN AMER. & CARIB',
    'Equatorial Guinea': 'SUB-SAHARAN AFRICA',
    'Eritrea': 'SUB-SAHARAN AFRICA',
    'Estonia': 'BALTICS',
    'Ethiopia': 'SUB-SAHARAN AFRICA',
    'Faroe Islands': 'WESTERN EUROPE',
    'Fiji': 'OCEANIA',
    'Finland': 'WESTERN EUROPE',
    'France': 'WESTERN EUROPE',
    'French Guiana': 'LATIN AMER. & CARIB',
    'French Polynesia': 'OCEANIA',
    'Gabon': 'SUB-SAHARAN AFRICA',
    'Gambia, The': 'SUB-SAHARAN AFRICA',
    'Gaza Strip': 'NEAR EAST',
    'Georgia': 'C.W. OF IND. STATES',
    'Germany': 'WESTERN EUROPE',
    'Ghana': 'SUB-SAHARAN AFRICA',
    'Gibraltar': 'WESTERN EUROPE',
    'Greece': 'WESTERN EUROPE',
    'Greenland': 'NORTHERN AMERICA',
    'Grenada': 'LATIN AMER. & CARIB',
    'Guadeloupe': 'LATIN AMER. & CARIB',
    'Guam': 'OCEANIA',
    'Guatemala': 'LATIN AMER. & CARIB',
    'Guernsey': 'WESTERN EUROPE',
    'Guinea': 'SUB-SAHARAN AFRICA',
    'Guinea-Bissau': 'SUB-SAHARAN AFRICA',
    'Guyana': 'LATIN AMER. & CARIB',
    'Haiti': 'LATIN AMER. & CARIB',
    'Honduras': 'LATIN AMER. & CARIB',
    'Hong Kong': 'ASIA (EX. NEAR EAST)',
    'Hungary': 'EASTERN EUROPE',
    'Iceland': 'WESTERN EUROPE',
    'India': 'ASIA (EX. NEAR EAST)',
    'Indonesia': 'ASIA (EX. NEAR EAST)',
    'Iran': 'ASIA (EX. NEAR EAST)',
    'Iraq': 'NEAR EAST',
    'Ireland': 'WESTERN EUROPE',
    'Isle of Man': 'WESTERN EUROPE',
    'Israel': 'NEAR EAST',
    'Italy': 'WESTERN EUROPE',
    'Jamaica': 'LATIN AMER. & CARIB',
    'Japan': 'ASIA (EX. NEAR EAST)',
    'Jersey': 'WESTERN EUROPE',
    'Jordan': 'NEAR EAST',
    'Kazakhstan': 'C.W. OF IND. STATES',
    'Kenya': 'SUB-SAHARAN AFRICA',
    'Kiribati': 'OCEANIA',
    'Korea, North': 'ASIA (EX. NEAR EAST)',
    'Korea, South': 'ASIA (EX. NEAR EAST)',
    'Kuwait': 'NEAR EAST',
    'Kyrgyzstan': 'C.W. OF IND. STATES',
    'Laos': 'ASIA (EX. NEAR EAST)',
    'Latvia': 'BALTICS',
    'Lebanon': 'NEAR EAST',
    'Lesotho': 'SUB-SAHARAN AFRICA',
    'Liberia': 'SUB-SAHARAN AFRICA',
    'Libya': 'NORTHERN AFRICA',
    'Liechtenstein': 'WESTERN EUROPE',
    'Lithuania': 'BALTICS',
    'Luxembourg': 'WESTERN EUROPE',
    'Macau': 'ASIA (EX. NEAR EAST)',
    'Macedonia': 'EASTERN EUROPE',
    'Madagascar': 'SUB-SAHARAN AFRICA',
    'Malawi': 'SUB-SAHARAN AFRICA',
    'Malaysia': 'ASIA (EX. NEAR EAST)',
    'Maldives': 'ASIA (EX. NEAR EAST)',
    'Mali': 'SUB-SAHARAN AFRICA',
    'Malta': 'WESTERN EUROPE',
    'Marshall Islands': 'OCEANIA',
    'Martinique': 'LATIN AMER. & CARIB',
    'Mauritania': 'SUB-SAHARAN AFRICA',
    'Mauritius': 'SUB-SAHARAN AFRICA',
    'Mayotte': 'SUB-SAHARAN AFRICA',
    'Mexico': 'LATIN AMER. & CARIB',
    'Micronesia, Fed. St.': 'OCEANIA',
    'Moldova': 'C.W. OF IND. STATES',
    'Monaco': 'WESTERN EUROPE',
    'Mongolia': 'ASIA (EX. NEAR EAST)',
    'Montserrat': 'LATIN AMER. & CARIB',
    'Morocco': 'NORTHERN AFRICA',
    'Mozambique': 'SUB-SAHARAN AFRICA',
    'Namibia': 'SUB-SAHARAN AFRICA',
    'Nauru': 'OCEANIA',
    'Nepal': 'ASIA (EX. NEAR EAST)',
    'Netherlands': 'WESTERN EUROPE',
    'Netherlands Antilles': 'LATIN AMER. & CARIB',
    'New Caledonia': 'OCEANIA',
    'New Zealand': 'OCEANIA',
    'Nicaragua': 'LATIN AMER. & CARIB',
    'Niger': 'SUB-SAHARAN AFRICA',
    'Nigeria': 'SUB-SAHARAN AFRICA',
    'N. Mariana Islands': 'OCEANIA',
    'Norway': 'WESTERN EUROPE',
    'Oman': 'NEAR EAST',
    'Pakistan': 'ASIA (EX. NEAR EAST)',
    'Palau': 'OCEANIA',
    'Panama': 'LATIN AMER. & CARIB',
    'Papua New Guinea': 'OCEANIA',
    'Paraguay': 'LATIN AMER. & CARIB',
    'Peru': 'LATIN AMER. & CARIB',
    'Philippines': 'ASIA (EX. NEAR EAST)',
    'Poland': 'EASTERN EUROPE',
    'Portugal': 'WESTERN EUROPE',
    'Puerto Rico': 'LATIN AMER. & CARIB',
    'Qatar': 'NEAR EAST',
    'Reunion': 'SUB-SAHARAN AFRICA',
    'Romania': 'EASTERN EUROPE',
    'Russia': 'C.W. OF IND. STATES',
    'Rwanda': 'SUB-SAHARAN AFRICA',
    'Saint Helena': 'SUB-SAHARAN AFRICA',
    'Saint Kitts & Nevis': 'LATIN AMER. & CARIB',
    'Saint Lucia': 'LATIN AMER. & CARIB',
    'St Pierre & Miquelon': 'NORTHERN AMERICA',
    'Saint Vincent and the Grenadines': 'LATIN AMER. & CARIB',
    'Samoa': 'OCEANIA',
    'San Marino': 'WESTERN EUROPE',
    'Sao Tome & Principe': 'SUB-SAHARAN AFRICA',
    'Saudi Arabia': 'NEAR EAST',
    'Senegal': 'SUB-SAHARAN AFRICA',
    'Serbia': 'EASTERN EUROPE',
    'Seychelles': 'SUB-SAHARAN AFRICA',
    'Sierra Leone': 'SUB-SAHARAN AFRICA',
    'Singapore': 'ASIA (EX. NEAR EAST)',
    'Slovakia': 'EASTERN EUROPE',
    'Slovenia': 'EASTERN EUROPE',
    'Solomon Islands': 'OCEANIA',
    'Somalia': 'SUB-SAHARAN AFRICA',
    'South Africa': 'SUB-SAHARAN AFRICA',
    'Spain': 'WESTERN EUROPE',
    'Sri Lanka': 'ASIA (EX. NEAR EAST)',
    'Sudan': 'SUB-SAHARAN AFRICA',
    'Suriname': 'LATIN AMER. & CARIB',
    'Swaziland': 'SUB-SAHARAN AFRICA',
    'Sweden': 'WESTERN EUROPE',
    'Switzerland': 'WESTERN EUROPE',
    'Syria': 'NEAR EAST',
    'Taiwan': 'ASIA (EX. NEAR EAST)',
    'Tajikistan': 'C.W. OF IND. STATES',
    'Tanzania': 'SUB-SAHARAN AFRICA',
    'Thailand': 'ASIA (EX. NEAR EAST)',
    'Togo': 'SUB-SAHARAN AFRICA',
    'Tonga': 'OCEANIA',
    'Trinidad & Tobago': 'LATIN AMER. & CARIB',
    'Tunisia': 'NORTHERN AFRICA',
    'Turkey': 'NEAR EAST',
    'Turkmenistan': 'C.W. OF IND. STATES',
    'Turks & Caicos Is': 'LATIN AMER. & CARIB',
    'Tuvalu': 'OCEANIA',
    'Uganda': 'SUB-SAHARAN AFRICA',
    'Ukraine': 'C.W. OF IND. STATES',
    'United Arab Emirates': 'NEAR EAST',
    'United Kingdom': 'WESTERN EUROPE',
    'United States': 'NORTHERN AMERICA',
    'Uruguay': 'LATIN AMER. & CARIB',
    'Uzbekistan': 'C.W. OF IND. STATES',
    'Vanuatu': 'OCEANIA',
    'Venezuela': 'LATIN AMER. & CARIB',
    'Vietnam': 'ASIA (EX. NEAR EAST)',
    'Virgin Islands': 'LATIN AMER. & CARIB',
    'Wallis and Futuna': 'OCEANIA',
    'West Bank': 'NEAR EAST',
    'Western Sahara': 'NORTHERN AFRICA',
    'Yemen': 'NEAR EAST',
    'Zambia': 'SUB-SAHARAN AFRICA',
    'Zimbabwe': 'SUB-SAHARAN AFRICA',
}

FACTS = {
    'Afghanistan': dict(capital='Kabul', currency='Afghan Afghani', language='Pashto, Dari',
        places=['Band-e-Amir National Park', 'Blue Mosque, Mazar-i-Sharif', 'Minaret of Jam']),
    'Albania': dict(capital='Tirana', currency='Albanian Lek', language='Albanian',
        places=['Berat Old Town', 'Albanian Riviera', 'Butrint National Park']),
    'Algeria': dict(capital='Algiers', currency='Algerian Dinar', language='Arabic',
        places=['Casbah of Algiers', 'Sahara Desert', 'Tassili n\'Ajjer']),
    'Andorra': dict(capital='Andorra la Vella', currency='Euro', language='Catalan',
        places=['Vallnord Ski Resort', 'Casa de la Vall', 'Madriu-Perafita-Claror Valley']),
    'Angola': dict(capital='Luanda', currency='Angolan Kwanza', language='Portuguese',
        places=['Kalandula Falls', 'Kissama National Park', 'Fortress of Sao Miguel']),
    'Argentina': dict(capital='Buenos Aires', currency='Argentine Peso', language='Spanish',
        places=['Iguazu Falls', 'Perito Moreno Glacier', 'Recoleta, Buenos Aires']),
    'Armenia': dict(capital='Yerevan', currency='Armenian Dram', language='Armenian',
        places=['Lake Sevan', 'Tatev Monastery', 'Garni Temple']),
    'Australia': dict(capital='Canberra', currency='Australian Dollar', language='English',
        places=['Sydney Opera House', 'Great Barrier Reef', 'Uluru']),
    'Austria': dict(capital='Vienna', currency='Euro', language='German',
        places=['Schonbrunn Palace', 'Hallstatt', 'Salzburg Old Town']),
    'Azerbaijan': dict(capital='Baku', currency='Azerbaijani Manat', language='Azerbaijani',
        places=['Flame Towers, Baku', 'Gobustan National Park', 'Sheki Khan Palace']),
    'Bahamas, The': dict(capital='Nassau', currency='Bahamian Dollar', language='English',
        places=['Exuma Cays', 'Pink Sands Beach', 'Nassau Straw Market']),
    'Bahrain': dict(capital='Manama', currency='Bahraini Dinar', language='Arabic',
        places=['Bahrain Fort', 'Bahrain World Trade Center', 'Tree of Life']),
    'Bangladesh': dict(capital='Dhaka', currency='Bangladeshi Taka', language='Bengali',
        places=['Sundarbans Mangrove Forest', 'Cox\'s Bazar Beach', 'Sixty Dome Mosque']),
    'Barbados': dict(capital='Bridgetown', currency='Barbadian Dollar', language='English',
        places=['Bathsheba Beach', 'Harrison\'s Cave', 'Historic Bridgetown']),
    'Belarus': dict(capital='Minsk', currency='Belarusian Ruble', language='Belarusian, Russian',
        places=['Mir Castle', 'Belovezhskaya Pushcha', 'Niasvizh Castle']),
    'Belgium': dict(capital='Brussels', currency='Euro', language='Dutch, French, German',
        places=['Grand Place, Brussels', 'Bruges Old Town', 'Atomium']),
    'Belize': dict(capital='Belmopan', currency='Belize Dollar', language='English',
        places=['Great Blue Hole', 'Belize Barrier Reef', 'Caracol Maya Ruins']),
    'Benin': dict(capital='Porto-Novo', currency='West African CFA Franc', language='French',
        places=['Royal Palaces of Abomey', 'Pendjari National Park', 'Ganvie Stilt Village']),
    'Bhutan': dict(capital='Thimphu', currency='Bhutanese Ngultrum', language='Dzongkha',
        places=['Paro Taktsang (Tiger\'s Nest)', 'Punakha Dzong', 'Buddha Dordenma']),
    'Bolivia': dict(capital='Sucre', currency='Bolivian Boliviano', language='Spanish',
        places=['Salar de Uyuni', 'Lake Titicaca', 'Tiwanaku Ruins']),
    'Bosnia & Herzegovina': dict(capital='Sarajevo', currency='Bosnia-Herzegovina Convertible Mark', language='Bosnian, Croatian, Serbian',
        places=['Stari Most, Mostar', 'Sarajevo Old Town', 'Kravice Waterfalls']),
    'Botswana': dict(capital='Gaborone', currency='Botswana Pula', language='English, Setswana',
        places=['Okavango Delta', 'Chobe National Park', 'Central Kalahari Game Reserve']),
    'Brazil': dict(capital='Brasilia', currency='Brazilian Real', language='Portuguese',
        places=['Christ the Redeemer', 'Amazon Rainforest', 'Iguazu Falls']),
    'Brunei': dict(capital='Bandar Seri Begawan', currency='Brunei Dollar', language='Malay',
        places=['Sultan Omar Ali Saifuddien Mosque', 'Kampong Ayer', 'Ulu Temburong National Park']),
    'Bulgaria': dict(capital='Sofia', currency='Bulgarian Lev', language='Bulgarian',
        places=['Rila Monastery', 'Plovdiv Old Town', 'Alexander Nevsky Cathedral']),
    'Burkina Faso': dict(capital='Ouagadougou', currency='West African CFA Franc', language='French',
        places=['Sindou Peaks', 'Ruins of Loropeni', 'W National Park']),
    'Burundi': dict(capital='Gitega', currency='Burundian Franc', language='Kirundi, French',
        places=['Lake Tanganyika', 'Rusizi National Park', 'Karera Falls']),
    'Cambodia': dict(capital='Phnom Penh', currency='Cambodian Riel', language='Khmer',
        places=['Angkor Wat', 'Bayon Temple', 'Tonle Sap Lake']),
    'Cameroon': dict(capital='Yaounde', currency='Central African CFA Franc', language='English, French',
        places=['Mount Cameroon', 'Waza National Park', 'Limbe Botanical Garden']),
    'Canada': dict(capital='Ottawa', currency='Canadian Dollar', language='English, French',
        places=['Niagara Falls', 'Banff National Park', 'CN Tower']),
    'Chad': dict(capital='N\'Djamena', currency='Central African CFA Franc', language='French, Arabic',
        places=['Zakouma National Park', 'Ennedi Plateau', 'Lake Chad']),
    'Chile': dict(capital='Santiago', currency='Chilean Peso', language='Spanish',
        places=['Atacama Desert', 'Torres del Paine', 'Easter Island']),
    'China': dict(capital='Beijing', currency='Chinese Yuan', language='Mandarin Chinese',
        places=['Great Wall of China', 'Forbidden City', 'Terracotta Army']),
    'Colombia': dict(capital='Bogota', currency='Colombian Peso', language='Spanish',
        places=['Cartagena Old Town', 'Tayrona National Park', 'Coffee Cultural Landscape']),
    'Congo, Dem. Rep.': dict(capital='Kinshasa', currency='Congolese Franc', language='French',
        places=['Virunga National Park', 'Congo River', 'Nyiragongo Volcano']),
    'Congo, Repub. of the': dict(capital='Brazzaville', currency='Central African CFA Franc', language='French',
        places=['Odzala-Kokoua National Park', 'Congo River Basin', 'Basilique Sainte-Anne']),
    'Costa Rica': dict(capital='San Jose', currency='Costa Rican Colon', language='Spanish',
        places=['Arenal Volcano', 'Manuel Antonio National Park', 'Monteverde Cloud Forest']),
    "Cote d'Ivoire": dict(capital='Yamoussoukro', currency='West African CFA Franc', language='French',
        places=['Basilica of Our Lady of Peace', 'Comoe National Park', 'Taï National Park']),
    'Croatia': dict(capital='Zagreb', currency='Euro', language='Croatian',
        places=['Dubrovnik Old Town', 'Plitvice Lakes', 'Diocletian\'s Palace']),
    'Cuba': dict(capital='Havana', currency='Cuban Peso', language='Spanish',
        places=['Old Havana', 'Vinales Valley', 'Trinidad Old Town']),
    'Cyprus': dict(capital='Nicosia', currency='Euro', language='Greek, Turkish',
        places=['Tombs of the Kings', 'Kyrenia Castle', 'Kourion Ancient Site']),
    'Czech Republic': dict(capital='Prague', currency='Czech Koruna', language='Czech',
        places=['Prague Castle', 'Charles Bridge', 'Cesky Krumlov']),
    'Denmark': dict(capital='Copenhagen', currency='Danish Krone', language='Danish',
        places=['Tivoli Gardens', 'Nyhavn', 'The Little Mermaid Statue']),
    'Djibouti': dict(capital='Djibouti City', currency='Djiboutian Franc', language='French, Arabic',
        places=['Lake Assal', 'Day Forest National Park', 'Moucha Island']),
    'Dominican Republic': dict(capital='Santo Domingo', currency='Dominican Peso', language='Spanish',
        places=['Punta Cana Beaches', 'Colonial Zone of Santo Domingo', 'Los Haitises National Park']),
    'Ecuador': dict(capital='Quito', currency='US Dollar', language='Spanish',
        places=['Galapagos Islands', 'Quito Old Town', 'Cotopaxi Volcano']),
    'Egypt': dict(capital='Cairo', currency='Egyptian Pound', language='Arabic',
        places=['Pyramids of Giza', 'Nile River', 'Valley of the Kings']),
    'El Salvador': dict(capital='San Salvador', currency='US Dollar', language='Spanish',
        places=['Joya de Ceren', 'El Boqueron National Park', 'Ruta de Las Flores']),
    'Estonia': dict(capital='Tallinn', currency='Euro', language='Estonian',
        places=['Tallinn Old Town', 'Lahemaa National Park', 'Kadriorg Palace']),
    'Ethiopia': dict(capital='Addis Ababa', currency='Ethiopian Birr', language='Amharic',
        places=['Lalibela Rock Churches', 'Simien Mountains', 'Danakil Depression']),
    'Fiji': dict(capital='Suva', currency='Fijian Dollar', language='English, Fijian',
        places=['Mamanuca Islands', 'Sigatoka Sand Dunes', 'Bouma National Heritage Park']),
    'Finland': dict(capital='Helsinki', currency='Euro', language='Finnish, Swedish',
        places=['Lapland', 'Suomenlinna Fortress', 'Northern Lights, Rovaniemi']),
    'France': dict(capital='Paris', currency='Euro', language='French',
        places=['Eiffel Tower', 'Louvre Museum', 'Palace of Versailles']),
    'Gabon': dict(capital='Libreville', currency='Central African CFA Franc', language='French',
        places=['Loango National Park', 'Ivindo National Park', 'Pongara National Park']),
    'Gambia, The': dict(capital='Banjul', currency='Gambian Dalasi', language='English',
        places=['Kunta Kinteh Island', 'Abuko Nature Reserve', 'River Gambia National Park']),
    'Georgia': dict(capital='Tbilisi', currency='Georgian Lari', language='Georgian',
        places=['Tbilisi Old Town', 'Svaneti Region', 'Gergeti Trinity Church']),
    'Germany': dict(capital='Berlin', currency='Euro', language='German',
        places=['Brandenburg Gate', 'Neuschwanstein Castle', 'Cologne Cathedral']),
    'Ghana': dict(capital='Accra', currency='Ghanaian Cedi', language='English',
        places=['Cape Coast Castle', 'Kakum National Park', 'Lake Volta']),
    'Greece': dict(capital='Athens', currency='Euro', language='Greek',
        places=['Acropolis of Athens', 'Santorini', 'Meteora Monasteries']),
    'Greenland': dict(capital='Nuuk', currency='Danish Krone', language='Greenlandic',
        places=['Ilulissat Icefjord', 'Disko Bay', 'Northern Lights']),
    'Guatemala': dict(capital='Guatemala City', currency='Guatemalan Quetzal', language='Spanish',
        places=['Tikal National Park', 'Lake Atitlan', 'Antigua Guatemala']),
    'Guinea': dict(capital='Conakry', currency='Guinean Franc', language='French',
        places=['Fouta Djallon', 'Mount Nimba', 'Iles de Los']),
    'Guyana': dict(capital='Georgetown', currency='Guyanese Dollar', language='English',
        places=['Kaieteur Falls', 'Iwokrama Rainforest', 'St. George\'s Cathedral']),
    'Haiti': dict(capital='Port-au-Prince', currency='Haitian Gourde', language='French, Haitian Creole',
        places=['Citadelle Laferriere', 'Sans-Souci Palace', 'Bassin Bleu']),
    'Honduras': dict(capital='Tegucigalpa', currency='Honduran Lempira', language='Spanish',
        places=['Copan Ruins', 'Roatan Island', 'Pico Bonito National Park']),
    'Hungary': dict(capital='Budapest', currency='Hungarian Forint', language='Hungarian',
        places=['Buda Castle', 'Szechenyi Thermal Bath', 'Fisherman\'s Bastion']),
    'Iceland': dict(capital='Reykjavik', currency='Icelandic Krona', language='Icelandic',
        places=['Blue Lagoon', 'Golden Circle', 'Jokulsarlon Glacier Lagoon']),
    'India': dict(capital='New Delhi', currency='Indian Rupee', language='Hindi, English',
        places=['Taj Mahal', 'Jaipur City Palace', 'Kerala Backwaters']),
    'Indonesia': dict(capital='Jakarta', currency='Indonesian Rupiah', language='Indonesian',
        places=['Borobudur Temple', 'Bali', 'Komodo National Park']),
    'Iran': dict(capital='Tehran', currency='Iranian Rial', language='Persian (Farsi)',
        places=['Persepolis', 'Naqsh-e Jahan Square, Isfahan', 'Golestan Palace']),
    'Iraq': dict(capital='Baghdad', currency='Iraqi Dinar', language='Arabic, Kurdish',
        places=['Ziggurat of Ur', 'Babylon Ruins', 'Erbil Citadel']),
    'Ireland': dict(capital='Dublin', currency='Euro', language='English, Irish',
        places=['Cliffs of Moher', 'Ring of Kerry', 'Trinity College Dublin']),
    'Israel': dict(capital='Jerusalem', currency='Israeli New Shekel', language='Hebrew',
        places=['Western Wall, Jerusalem', 'Dead Sea', 'Old City of Jerusalem']),
    'Italy': dict(capital='Rome', currency='Euro', language='Italian',
        places=['Colosseum', 'Venice Canals', 'Leaning Tower of Pisa']),
    'Jamaica': dict(capital='Kingston', currency='Jamaican Dollar', language='English',
        places=['Dunn\'s River Falls', 'Blue Mountains', 'Seven Mile Beach, Negril']),
    'Japan': dict(capital='Tokyo', currency='Japanese Yen', language='Japanese',
        places=['Mount Fuji', 'Kyoto Old Town', 'Tokyo Tower']),
    'Jordan': dict(capital='Amman', currency='Jordanian Dinar', language='Arabic',
        places=['Petra', 'Wadi Rum', 'Dead Sea']),
    'Kazakhstan': dict(capital='Astana', currency='Kazakhstani Tenge', language='Kazakh, Russian',
        places=['Charyn Canyon', 'Baiterek Tower', 'Big Almaty Lake']),
    'Kenya': dict(capital='Nairobi', currency='Kenyan Shilling', language='Swahili, English',
        places=['Maasai Mara National Reserve', 'Mount Kenya', 'Amboseli National Park']),
    'Kiribati': dict(capital='Tarawa', currency='Australian Dollar', language='English, Gilbertese',
        places=['Tarawa Atoll', 'Phoenix Islands Protected Area', 'Christmas Island']),
    'Kuwait': dict(capital='Kuwait City', currency='Kuwaiti Dinar', language='Arabic',
        places=['Kuwait Towers', 'Grand Mosque of Kuwait', 'Failaka Island']),
    'Kyrgyzstan': dict(capital='Bishkek', currency='Kyrgyzstani Som', language='Kyrgyz, Russian',
        places=['Lake Issyk-Kul', 'Ala Archa National Park', 'Osh Bazaar']),
    'Laos': dict(capital='Vientiane', currency='Lao Kip', language='Lao',
        places=['Luang Prabang', 'Plain of Jars', 'Kuang Si Falls']),
    'Latvia': dict(capital='Riga', currency='Euro', language='Latvian',
        places=['Riga Old Town', 'Gauja National Park', 'Rundale Palace']),
    'Lebanon': dict(capital='Beirut', currency='Lebanese Pound', language='Arabic',
        places=['Baalbek Ruins', 'Jeita Grotto', 'Byblos']),
    'Lesotho': dict(capital='Maseru', currency='Lesotho Loti', language='Sesotho, English',
        places=['Sani Pass', 'Maletsunyane Falls', 'Thaba Bosiu']),
    'Liberia': dict(capital='Monrovia', currency='Liberian Dollar', language='English',
        places=['Sapo National Park', 'Providence Island', 'Robertsport Beach']),
    'Libya': dict(capital='Tripoli', currency='Libyan Dinar', language='Arabic',
        places=['Leptis Magna', 'Sabratha Ruins', 'Ghadames Old Town']),
    'Liechtenstein': dict(capital='Vaduz', currency='Swiss Franc', language='German',
        places=['Vaduz Castle', 'Malbun Ski Resort', 'Gutenberg Castle']),
    'Lithuania': dict(capital='Vilnius', currency='Euro', language='Lithuanian',
        places=['Vilnius Old Town', 'Curonian Spit', 'Trakai Island Castle']),
    'Luxembourg': dict(capital='Luxembourg City', currency='Euro', language='Luxembourgish, French, German',
        places=['Luxembourg City Old Quarters', 'Vianden Castle', 'Mullerthal Trail']),
    'Macedonia': dict(capital='Skopje', currency='Macedonian Denar', language='Macedonian',
        places=['Lake Ohrid', 'Matka Canyon', 'Skopje Old Bazaar']),
    'Madagascar': dict(capital='Antananarivo', currency='Malagasy Ariary', language='Malagasy, French',
        places=['Avenue of the Baobabs', 'Tsingy de Bemaraha', 'Andasibe-Mantadia National Park']),
    'Malawi': dict(capital='Lilongwe', currency='Malawian Kwacha', language='English, Chichewa',
        places=['Lake Malawi', 'Nyika National Park', 'Zomba Plateau']),
    'Malaysia': dict(capital='Kuala Lumpur', currency='Malaysian Ringgit', language='Malay',
        places=['Petronas Twin Towers', 'Langkawi Islands', 'Batu Caves']),
    'Maldives': dict(capital='Male', currency='Maldivian Rufiyaa', language='Dhivehi',
        places=['Male Atoll', 'Baa Atoll Biosphere Reserve', 'Overwater Villas, Ari Atoll']),
    'Mali': dict(capital='Bamako', currency='West African CFA Franc', language='French',
        places=['Great Mosque of Djenne', 'Timbuktu', 'Bandiagara Escarpment']),
    'Malta': dict(capital='Valletta', currency='Euro', language='Maltese, English',
        places=['Valletta Old Town', 'Blue Lagoon, Comino', 'Mdina']),
    'Mauritania': dict(capital='Nouakchott', currency='Mauritanian Ouguiya', language='Arabic',
        places=['Banc d\'Arguin National Park', 'Chinguetti', 'Richat Structure']),
    'Mauritius': dict(capital='Port Louis', currency='Mauritian Rupee', language='English, French, Creole',
        places=['Le Morne Brabant', 'Black River Gorges National Park', 'Chamarel Seven Coloured Earths']),
    'Mexico': dict(capital='Mexico City', currency='Mexican Peso', language='Spanish',
        places=['Chichen Itza', 'Teotihuacan', 'Cancun Beaches']),
    'Micronesia, Fed. St.': dict(capital='Palikir', currency='US Dollar', language='English',
        places=['Nan Madol Ruins', 'Chuuk Lagoon', 'Kosrae Island']),
    'Moldova': dict(capital='Chisinau', currency='Moldovan Leu', language='Romanian',
        places=['Orheiul Vechi', 'Cricova Winery', 'Soroca Fortress']),
    'Monaco': dict(capital='Monaco', currency='Euro', language='French',
        places=['Monte Carlo Casino', 'Prince\'s Palace of Monaco', 'Monaco Grand Prix Circuit']),
    'Mongolia': dict(capital='Ulaanbaatar', currency='Mongolian Togrog', language='Mongolian',
        places=['Gobi Desert', 'Erdene Zuu Monastery', 'Khustain Nuruu National Park']),
    'Morocco': dict(capital='Rabat', currency='Moroccan Dirham', language='Arabic',
        places=['Chefchaouen', 'Sahara Desert', 'Marrakech Medina']),
    'Mozambique': dict(capital='Maputo', currency='Mozambican Metical', language='Portuguese',
        places=['Bazaruto Archipelago', 'Gorongosa National Park', 'Island of Mozambique']),
    'Myanmar': dict(capital='Naypyidaw', currency='Myanmar Kyat', language='Burmese',
        places=['Bagan Temples', 'Inle Lake', 'Shwedagon Pagoda']),
    'Namibia': dict(capital='Windhoek', currency='Namibian Dollar', language='English',
        places=['Sossusvlei Dunes', 'Etosha National Park', 'Fish River Canyon']),
    'Nepal': dict(capital='Kathmandu', currency='Nepalese Rupee', language='Nepali',
        places=['Mount Everest', 'Kathmandu Durbar Square', 'Pokhara']),
    'Netherlands': dict(capital='Amsterdam', currency='Euro', language='Dutch',
        places=['Keukenhof Gardens', 'Amsterdam Canals', 'Windmills of Kinderdijk']),
    'New Zealand': dict(capital='Wellington', currency='New Zealand Dollar', language='English, Maori',
        places=['Milford Sound', 'Hobbiton Movie Set', 'Rotorua Geothermal Park']),
    'Nicaragua': dict(capital='Managua', currency='Nicaraguan Cordoba', language='Spanish',
        places=['Ometepe Island', 'Granada Colonial City', 'Masaya Volcano']),
    'Niger': dict(capital='Niamey', currency='West African CFA Franc', language='French',
        places=['W National Park', 'Air Mountains', 'Grand Marche de Niamey']),
    'Nigeria': dict(capital='Abuja', currency='Nigerian Naira', language='English',
        places=['Zuma Rock', 'Yankari National Park', 'Olumo Rock']),
    'North Korea': dict(capital='Pyongyang', currency='North Korean Won', language='Korean',
        places=['Kumsusan Palace', 'Mount Paektu', 'Juche Tower']),
    'Norway': dict(capital='Oslo', currency='Norwegian Krone', language='Norwegian',
        places=['Geirangerfjord', 'Northern Lights, Tromso', 'Bryggen, Bergen']),
    'Oman': dict(capital='Muscat', currency='Omani Rial', language='Arabic',
        places=['Sultan Qaboos Grand Mosque', 'Wahiba Sands', 'Jebel Shams']),
    'Pakistan': dict(capital='Islamabad', currency='Pakistani Rupee', language='Urdu, English',
        places=['Badshahi Mosque', 'Hunza Valley', 'Faisal Mosque']),
    'Palau': dict(capital='Ngerulmud', currency='US Dollar', language='Palauan, English',
        places=['Rock Islands', 'Jellyfish Lake', 'Milky Way Lagoon']),
    'Panama': dict(capital='Panama City', currency='Panamanian Balboa', language='Spanish',
        places=['Panama Canal', 'Casco Viejo', 'San Blas Islands']),
    'Papua New Guinea': dict(capital='Port Moresby', currency='Papua New Guinean Kina', language='English, Tok Pisin',
        places=['Kokoda Track', 'Sepik River', 'Mount Wilhelm']),
    'Paraguay': dict(capital='Asuncion', currency='Paraguayan Guarani', language='Spanish, Guarani',
        places=['Jesuit Missions of La Santisima Trinidad', 'Itaipu Dam', 'Ybycui National Park']),
    'Peru': dict(capital='Lima', currency='Peruvian Sol', language='Spanish',
        places=['Machu Picchu', 'Nazca Lines', 'Sacred Valley']),
    'Philippines': dict(capital='Manila', currency='Philippine Peso', language='Filipino, English',
        places=['Chocolate Hills', 'Palawan Islands', 'Banaue Rice Terraces']),
    'Poland': dict(capital='Warsaw', currency='Polish Zloty', language='Polish',
        places=['Krakow Old Town', 'Wieliczka Salt Mine', 'Malbork Castle']),
    'Portugal': dict(capital='Lisbon', currency='Euro', language='Portuguese',
        places=['Belem Tower', 'Porto Old Town', 'Algarve Coast']),
    'Qatar': dict(capital='Doha', currency='Qatari Riyal', language='Arabic',
        places=['Museum of Islamic Art', 'The Pearl-Qatar', 'Souq Waqif']),
    'Romania': dict(capital='Bucharest', currency='Romanian Leu', language='Romanian',
        places=['Bran Castle', 'Peles Castle', 'Transfagarasan Highway']),
    'Russia': dict(capital='Moscow', currency='Russian Ruble', language='Russian',
        places=['Red Square', 'Hermitage Museum', 'Lake Baikal']),
    'Rwanda': dict(capital='Kigali', currency='Rwandan Franc', language='Kinyarwanda, English, French',
        places=['Volcanoes National Park', 'Lake Kivu', 'Nyungwe Forest National Park']),
    'Saint Lucia': dict(capital='Castries', currency='East Caribbean Dollar', language='English',
        places=['The Pitons', 'Sulphur Springs', 'Marigot Bay']),
    'Samoa': dict(capital='Apia', currency='Samoan Tala', language='Samoan, English',
        places=['To Sua Ocean Trench', 'Lalomanu Beach', 'Papase\'ea Sliding Rocks']),
    'Saudi Arabia': dict(capital='Riyadh', currency='Saudi Riyal', language='Arabic',
        places=['Masjid al-Haram, Mecca', 'Al-Ula / Hegra', 'Edge of the World']),
    'Senegal': dict(capital='Dakar', currency='West African CFA Franc', language='French',
        places=['Goree Island', 'Lake Retba (Pink Lake)', 'Djoudj National Bird Sanctuary']),
    'Serbia': dict(capital='Belgrade', currency='Serbian Dinar', language='Serbian',
        places=['Belgrade Fortress', 'Novi Sad', 'Studenica Monastery']),
    'Seychelles': dict(capital='Victoria', currency='Seychellois Rupee', language='Seychellois Creole, English, French',
        places=['Anse Source d\'Argent', 'Vallee de Mai', 'Aldabra Atoll']),
    'Sierra Leone': dict(capital='Freetown', currency='Sierra Leonean Leone', language='English',
        places=['Tacugama Chimpanzee Sanctuary', 'Banana Islands', 'Bunce Island']),
    'Singapore': dict(capital='Singapore', currency='Singapore Dollar', language='English, Malay, Mandarin, Tamil',
        places=['Marina Bay Sands', 'Gardens by the Bay', 'Sentosa Island']),
    'Slovakia': dict(capital='Bratislava', currency='Euro', language='Slovak',
        places=['Bratislava Castle', 'High Tatras', 'Spis Castle']),
    'Slovenia': dict(capital='Ljubljana', currency='Euro', language='Slovenian',
        places=['Lake Bled', 'Postojna Cave', 'Triglav National Park']),
    'Solomon Islands': dict(capital='Honiara', currency='Solomon Islands Dollar', language='English',
        places=['Marovo Lagoon', 'Guadalcanal', 'Tetepare Island']),
    'Somalia': dict(capital='Mogadishu', currency='Somali Shilling', language='Somali, Arabic',
        places=['Laas Geel Cave Paintings', 'Lido Beach, Mogadishu', 'Hargeisa']),
    'South Africa': dict(capital='Pretoria', currency='South African Rand', language='English, Zulu, Xhosa, Afrikaans',
        places=['Table Mountain', 'Kruger National Park', 'Cape of Good Hope']),
    'South Korea': dict(capital='Seoul', currency='South Korean Won', language='Korean',
        places=['Gyeongbokgung Palace', 'Jeju Island', 'Bukchon Hanok Village']),
    'Spain': dict(capital='Madrid', currency='Euro', language='Spanish',
        places=['Sagrada Familia', 'Alhambra', 'Park Guell']),
    'Sri Lanka': dict(capital='Sri Jayawardenepura Kotte', currency='Sri Lankan Rupee', language='Sinhala, Tamil',
        places=['Sigiriya Rock Fortress', 'Temple of the Tooth, Kandy', 'Yala National Park']),
    'Sudan': dict(capital='Khartoum', currency='Sudanese Pound', language='Arabic, English',
        places=['Pyramids of Meroe', 'Sanganeb Marine National Park', 'Jebel Barkal']),
    'Suriname': dict(capital='Paramaribo', currency='Surinamese Dollar', language='Dutch',
        places=['Historic Paramaribo', 'Central Suriname Nature Reserve', 'Brownsberg Nature Park']),
    'Swaziland': dict(capital='Mbabane', currency='Swazi Lilangeni', language='Swati, English',
        places=['Hlane Royal National Park', 'Mlilwane Wildlife Sanctuary', 'Ezulwini Valley']),
    'Sweden': dict(capital='Stockholm', currency='Swedish Krona', language='Swedish',
        places=['Vasa Museum, Stockholm', 'Gamla Stan', 'Icehotel, Jukkasjarvi']),
    'Switzerland': dict(capital='Bern', currency='Swiss Franc', language='German, French, Italian',
        places=['Matterhorn', 'Lake Lucerne', 'Jungfraujoch']),
    'Syria': dict(capital='Damascus', currency='Syrian Pound', language='Arabic',
        places=['Palmyra Ruins', 'Krak des Chevaliers', 'Old City of Damascus']),
    'Taiwan': dict(capital='Taipei', currency='New Taiwan Dollar', language='Mandarin Chinese',
        places=['Taipei 101', 'Taroko Gorge', 'National Palace Museum']),
    'Tajikistan': dict(capital='Dushanbe', currency='Tajikistani Somoni', language='Tajik',
        places=['Pamir Highway', 'Iskanderkul Lake', 'Fann Mountains']),
    'Tanzania': dict(capital='Dodoma', currency='Tanzanian Shilling', language='Swahili, English',
        places=['Mount Kilimanjaro', 'Serengeti National Park', 'Zanzibar']),
    'Thailand': dict(capital='Bangkok', currency='Thai Baht', language='Thai',
        places=['Grand Palace, Bangkok', 'Phi Phi Islands', 'Chiang Mai Old City']),
    'Togo': dict(capital='Lome', currency='West African CFA Franc', language='French',
        places=['Koutammakou (Land of the Batammariba)', 'Lake Togo', 'Fazao-Malfakassa National Park']),
    'Tonga': dict(capital="Nuku'alofa", currency='Tongan Pa\'anga', language='Tongan, English',
        places=['Ha\'amonga \'a Maui Trilithon', 'Swallows Cave', 'Mapu\'a \'a Vaea Blowholes']),
    'Trinidad & Tobago': dict(capital='Port of Spain', currency='Trinidad and Tobago Dollar', language='English',
        places=['Pitch Lake', 'Maracas Beach', 'Caroni Bird Sanctuary']),
    'Tunisia': dict(capital='Tunis', currency='Tunisian Dinar', language='Arabic',
        places=['Medina of Tunis', 'Sahara Desert (Douz)', 'Amphitheatre of El Jem']),
    'Turkey': dict(capital='Ankara', currency='Turkish Lira', language='Turkish',
        places=['Hagia Sophia', 'Cappadocia', 'Pamukkale']),
    'Turkmenistan': dict(capital='Ashgabat', currency='Turkmenistani Manat', language='Turkmen',
        places=['Darvaza Gas Crater (Door to Hell)', 'Ashgabat White Marble City', 'Ancient Merv']),
    'Uganda': dict(capital='Kampala', currency='Ugandan Shilling', language='English, Swahili',
        places=['Bwindi Impenetrable Forest', 'Murchison Falls National Park', 'Source of the Nile']),
    'Ukraine': dict(capital='Kyiv', currency='Ukrainian Hryvnia', language='Ukrainian',
        places=['Saint Sophia Cathedral, Kyiv', 'Lviv Old Town', 'Chernobyl Exclusion Zone']),
    'United Arab Emirates': dict(capital='Abu Dhabi', currency='UAE Dirham', language='Arabic',
        places=['Burj Khalifa', 'Sheikh Zayed Grand Mosque', 'Palm Jumeirah']),
    'United Kingdom': dict(capital='London', currency='British Pound', language='English',
        places=['Big Ben', 'Stonehenge', 'Buckingham Palace']),
    'United States': dict(capital='Washington, D.C.', currency='US Dollar', language='English',
        places=['Statue of Liberty', 'Grand Canyon', 'Yellowstone National Park']),
    'Uruguay': dict(capital='Montevideo', currency='Uruguayan Peso', language='Spanish',
        places=['Colonia del Sacramento', 'Punta del Este', 'Montevideo Old Town']),
    'Uzbekistan': dict(capital='Tashkent', currency='Uzbekistani Som', language='Uzbek',
        places=['Registan Square, Samarkand', 'Bukhara Old City', 'Khiva Old Town']),
    'Vanuatu': dict(capital='Port Vila', currency='Vanuatu Vatu', language='Bislama, English, French',
        places=['Mount Yasur Volcano', 'Blue Lagoon, Efate', 'Champagne Beach']),
    'Venezuela': dict(capital='Caracas', currency='Venezuelan Bolivar', language='Spanish',
        places=['Angel Falls', 'Los Roques Archipelago', 'Mount Roraima']),
    'Vietnam': dict(capital='Hanoi', currency='Vietnamese Dong', language='Vietnamese',
        places=['Ha Long Bay', 'Hoi An Ancient Town', 'Cu Chi Tunnels']),
    'Yemen': dict(capital="Sana'a", currency='Yemeni Rial', language='Arabic',
        places=['Old City of Sana\'a', 'Socotra Island', 'Shibam Old City']),
    'Zambia': dict(capital='Lusaka', currency='Zambian Kwacha', language='English',
        places=['Victoria Falls', 'South Luangwa National Park', 'Lake Kariba']),
    'Zimbabwe': dict(capital='Harare', currency='Zimbabwean Dollar', language='English, Shona',
        places=['Victoria Falls', 'Great Zimbabwe Ruins', 'Hwange National Park']),
    'Bhutan': dict(capital='Thimphu', currency='Bhutanese Ngultrum', language='Dzongkha',
        places=['Paro Taktsang', 'Punakha Dzong', 'Buddha Dordenma']),
}


# Countries where the CSV's population dataset uses a different official name
ALIASES = {
    'Myanmar': 'Burma',
    'North Korea': 'Korea, North',
    'South Korea': 'Korea, South',
}


def get_population(country: str):
    key = ALIASES.get(country, country)
    val = POPULATION.get(key)
    if val is None:
        return 'Data not available'
    return f'{val:,}'


def get_region(country: str):
    key = ALIASES.get(country, country)
    return REGION.get(key, 'Unknown region').title()


COUNTRIES = sorted(FACTS.keys())

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title='Country Explorer',
    page_icon='🌍',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ----------------------------------------------------------------------------
# PREMIUM CSS THEME
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 10% -10%, rgba(123,241,255,0.16) 0%, transparent 40%),
        radial-gradient(circle at 90% 0%, rgba(244,114,182,0.14) 0%, transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(167,139,250,0.18) 0%, transparent 50%),
        linear-gradient(180deg, #0b0a1f 0%, #100c26 60%, #0b0a1f 100%);
    background-attachment: fixed;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2.6rem; max-width: 980px; }

/* Floating decorative blobs */
.blob {
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.35;
    z-index: 0;
    animation: floaty 10s ease-in-out infinite;
}
@keyframes floaty {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-25px); }
}

.hero-wrap { text-align: center; margin-bottom: 2rem; }
.hero-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    background: rgba(123,241,255,0.1);
    border: 1px solid rgba(123,241,255,0.3);
    color: #7bf1ff;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7bf1ff 0%, #a78bfa 45%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-sub {
    color: #a9a6c9;
    font-size: 1.1rem;
    font-weight: 400;
    max-width: 620px;
    margin: 0 auto;
}

/* Search card */
.search-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 22px;
    padding: 1.8rem 2rem 1.4rem 2rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    margin-bottom: 1.6rem;
}

.stat-pill {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    color: #d5d2f5;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-right: 0.6rem;
    margin-top: 0.6rem;
    font-weight: 500;
}

/* Profile grid cards */
.field-card {
    background: linear-gradient(150deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.5rem 1.6rem;
    height: 100%;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    animation: cardIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.field-card:hover {
    transform: translateY(-4px);
    border-color: rgba(167,139,250,0.5);
    box-shadow: 0 16px 40px rgba(123,241,255,0.15);
}
.field-icon-wrap {
    width: 46px; height: 46px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    background: linear-gradient(135deg, rgba(123,241,255,0.18), rgba(167,139,250,0.18));
    margin-bottom: 0.9rem;
}
.field-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #8be9fd;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.field-value {
    font-size: 1.35rem;
    color: #f5f3ff;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1.35;
}
.field-value.places { font-size: 1rem; font-weight: 500; color: #e4e1ff; }

@keyframes cardIn {
    from { opacity: 0; transform: translateY(22px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.hero-country {
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.4rem 0 0.1rem 0;
}
.hero-region {
    text-align: center;
    color: #9c98c4;
    font-size: 0.95rem;
    margin-bottom: 1.6rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #7bf1ff, #a78bfa, #f472b6);
    background-size: 200% auto;
    color: #0b0a1f;
    font-weight: 700;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1.6rem;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.3px;
    font-size: 1rem;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 8px 24px rgba(167,139,250,0.35);
}
div.stButton > button:hover {
    background-position: right center;
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(167,139,250,0.5);
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.14);
}

.placeholder-box {
    text-align: center;
    padding: 3rem 2rem;
    color: #8a87ad;
    font-size: 1rem;
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 20px;
    background: rgba(255,255,255,0.02);
}
.placeholder-box .big-emoji { font-size: 2.6rem; display: block; margin-bottom: 0.6rem; }
</style>

<div class="blob" style="width:340px; height:340px; top:-100px; left:-120px; background:#7bf1ff;"></div>
<div class="blob" style="width:380px; height:380px; top:-60px; right:-140px; background:#f472b6; animation-delay: 2s;"></div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-badge">✨ Instant Country Profiles</div>'
    '<div class="hero-title">Country Explorer</div>'
    '<div class="hero-sub">Pick any country and get a beautifully curated snapshot — '
    'capital, currency, population, language and must-see landmarks.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SEARCH CARD
# ----------------------------------------------------------------------------
st.markdown('<div class="search-card">', unsafe_allow_html=True)
col_in, col_btn = st.columns([3, 1], vertical_alignment='bottom')
with col_in:
    country = st.selectbox(
        'Choose a country',
        COUNTRIES,
        index=COUNTRIES.index('Japan') if 'Japan' in COUNTRIES else 0,
        label_visibility='collapsed',
    )
with col_btn:
    generate = st.button('✨ Explore', use_container_width=True)

st.markdown(
    f'<span class="stat-pill">🗺️ {get_region(country)}</span>'
    f'<span class="stat-pill">👥 {get_population(country)}</span>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------------
output_slot = st.empty()

FIELDS = [
    ('🏛️', 'Capital', lambda d: d['capital'], False),
    ('💰', 'Currency', lambda d: d['currency'], False),
    ('👥', 'Population', lambda d: get_population(country), False),
    ('🗣️', 'Official Language', lambda d: d['language'], False),
    ('📍', 'Famous Places', lambda d: '<br>'.join(d['places']), True),
]


def render_profile(country_name, data, animate=True):
    rows_html = f'<div class="hero-country">{country_name}</div>'
    rows_html += f'<div class="hero-region">{get_region(country_name)}</div>'
    rows_html += '<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">'
    for i, (icon, label, getter, is_places) in enumerate(FIELDS):
        value = getter(data)
        delay = f'animation-delay: {i * 0.09:.2f}s;' if animate else ''
        span = 'grid-column: span 3;' if is_places else 'grid-column: span 1;'
        value_cls = 'field-value places' if is_places else 'field-value'
        rows_html += (
            f'<div class="field-card" style="{delay}{span}">'
            f'<div class="field-icon-wrap">{icon}</div>'
            f'<div class="field-label">{label}</div>'
            f'<div class="{value_cls}">{value}</div>'
            f'</div>'
        )
    rows_html += '</div>'
    return rows_html


if generate:
    data = FACTS[country]
    with st.spinner('Curating profile...'):
        time.sleep(0.5)
    output_slot.markdown(render_profile(country, data), unsafe_allow_html=True)
else:
    output_slot.markdown(
        '<div class="placeholder-box">'
        '<span class="big-emoji">🌐</span>'
        'Select a country above and hit <b>Explore</b> to reveal its profile.'
        '</div>',
        unsafe_allow_html=True,
    )
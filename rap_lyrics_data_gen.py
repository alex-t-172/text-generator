# -*- coding: utf-8 -*-

import requests
import bs4

def generate_ohhla_text():
    artist = input('Type name of artist to download text file for: ')

    main_page = requests.get('http://www.ohhla.com/YFA_{}.html'.format(artist))
    main_page.raise_for_status()
    main_page = bs4.BeautifulSoup(main_page.text)


    elems = main_page.select('a')
    filtered = []
    refiltered = []

    for i in elems:
        try:
            dic = i.attrs
            end_string = dic['href']
            if end_string[-4:] == '.txt':
                web_string = 'http://www.ohhla.com/' + end_string
                filtered.append(web_string)
        except:
            continue
    
    for i in filtered:
        if artist in i:
            refiltered.append(i)

    data_file = open('{}_data.txt'.format(artist), 'w')


    for song in refiltered:
        try:
            res = requests.get(song)
            res.raise_for_status()
            text = res.text
            filter_text = bs4.BeautifulSoup(text)
            elem_txt = filter_text.select('pre')
            elem_txt = elem_txt[0].getText()
        
            elem_txt = elem_txt.rpartition('Typed by: ')[-1]
            #elem_txt = elem_txt.rpartition('\\n')[-1]
            elem_txt = elem_txt.splitlines()[2:]
            elem_txt = '\n'.join(elem_txt)

            data_file.write(elem_txt)
            print(str(song) + ' written')
        except:
            print(str(song) + ' not downloaded')
    data_file.close()
    
    print()
    print('Text data file for {} generated'.format(artist))
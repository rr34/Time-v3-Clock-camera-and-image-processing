import os, shutil, string
import re
import numpy as np
import pandas as pd
import formatters


def readXMPfiles(XMPdirectory, columns_to_interpolate):
    XMPdictionary = {}
    XMP_list = []
    # read all the XMP text into a dictionary of strings and also make a list for the rows index of the dataframe
    for XMPfile in os.listdir(XMPdirectory):
        if XMPfile.endswith('.xmp'):
            XMPfullpath = os.path.join(XMPdirectory, XMPfile)
            XMPbase = os.path.splitext(os.path.basename(XMPfile))[0]
            XMP_list.append(XMPbase)
            print('Reading ' + XMPbase)
            with open(XMPfullpath, 'r') as f:
                XMPdictionary[XMPbase] = f.read()


    # using one of the XMP files, get the names of all the columns of data contained in the XMPs
    first_xmp = XMPdictionary[next(iter(XMPdictionary))]
    search_exif = r'(?<=exif:)[a-zA-Z\d]+(?==)'
    search_crs = r'(?<=crs:)[a-zA-Z\d]+(?==)'
    exif_list = re.findall(search_exif, first_xmp)
    exif_list = ['exif ' + s for s in exif_list]
    crs_list = re.findall(search_crs, first_xmp)
    crs_list = ['crs ' + s for s in crs_list]
    awim_list = ['awim CommaSeparatedTags']

    all_columns = list(set(exif_list + crs_list + awim_list + columns_to_interpolate))
    xmp_snapshot = pd.DataFrame(columns=all_columns, index=XMP_list)
    
    for key, value in XMPdictionary.items():
        print('Scrubbing ' + key)
        for column in xmp_snapshot:
            search_values = column.split(' ')
            if search_values[0] == 'crs' or search_values[0] == 'exif':
                search_re = r'(?<=%s:%s=")([^"])*(?=")' % (search_values[0], search_values[1])
                single_value = re.search(search_re, value)
                if single_value:
                    single_value = single_value.group()
                else:
                    single_value = 'No value found.'
            elif search_values[0] == 'awim' and search_values[1] == 'CommaSeparatedTags':
                search_re = r'(?<=<dc:subject>).*(?=</dc:subject>)'
                tags_fulltext = re.search(search_re, value, re.DOTALL)
                if tags_fulltext:
                    search_re = r'(?<=rdf:li>)[a-zA-Z\d]+(?=</rdf:li>)'
                    tags = re.findall(search_re, tags_fulltext.group())
                    if tags:
                        single_value = ','.join(tags)
                    else: single_value = 'placeholder'
                else:
                    single_value = 'placeholder'

            xmp_snapshot.loc[key,column] = single_value

    xmp_snapshot['exif DateTimeOriginal'] = formatters.format_datetime(xmp_snapshot['exif DateTimeOriginal'], 'ISO 8601 string tz to Zulu')
    xmp_snapshot = xmp_snapshot.sort_values('exif DateTimeOriginal')
    xmp_snapshot.insert(0, 'awim FrameNumber', range(0, len(xmp_snapshot)))

    if 'exif GPSLatitude' in xmp_snapshot.iloc[0]:
        latitude_txt_split = xmp_snapshot.iloc[0]['exif GPSLatitude'].split(',')
        latitude_hemisphere = latitude_txt_split[1][-1]
        latitude_txt_split[1] = latitude_txt_split[1][:-1]
        latitude_hemisphere_value = 1 if latitude_hemisphere == 'N' else -1
        latitude_value = latitude_hemisphere_value * (float(latitude_txt_split[0]) + float(latitude_txt_split[1])/60)
        longitude_txt_split = xmp_snapshot.iloc[0]['exif GPSLongitude'].split(',')
        longitude_hemisphere = longitude_txt_split[1][-1]
        longitude_txt_split[1] = longitude_txt_split[1][:-1]
        longitude_hemisphere_value = 1 if longitude_hemisphere == 'N' else -1
        longitude_value = longitude_hemisphere_value * (float(longitude_txt_split[0]) + float(longitude_txt_split[1])/60)
    else:
        latitude_value = 40.2987
        longitude_value = -83.0680

    latlng = [latitude_value, longitude_value]
    
    return xmp_snapshot, latlng


# todo: Unique tags only. Prevent this from adding duplicate tags.
def addTags(df_before, df_new, xmp_directory):
    for row, value in df_new.iterrows(): # 
        tags_before = df_before.loc[row][['awim CommaSeparatedTags']].values[0].split(',')
        tags_new = value[['awim CommaSeparatedTags']].values[0].split(',')
        tags_all = list(set(tags_before + tags_new))

        xmpfullpath = os.path.join(xmp_directory, row) + '.xmp'
        with open(xmpfullpath, 'r') as f:
            xmptext = f.read()
        if '<dc:subject>' in xmptext:
            xmptext = re.sub(r'\n<dc:subject>.*</dc:subject>', '', xmptext, flags=re.DOTALL)
        
        addition = '\n   <dc:subject>\n    <rdf:Bag>'
        for tag in tags_all:
            if tag != 'Placeholder':
                addition += '\n     <rdf:li>' + tag + '</rdf:li>'
        addition += '\n    </rdf:Bag>\n   </dc:subject>'

        xmptext = re.sub(r'(</xmpMM:History>)', rf'\1{addition}', xmptext)
        with open (xmpfullpath, 'w') as f:
            f.write(xmptext)


def interpolate(df_xmp, columns_to_interpolate):
    # set all the non-keyframe values equal to np.nan
    df_xmp.loc[~df_xmp['awim CommaSeparatedTags'].str.contains('keyframe', case=False), columns_to_interpolate] = np.nan
    # interpolate all the np.nan in between the keyframes, and extend the keyframe values backward to beginning and forward to the end
    df_xmp[columns_to_interpolate] = df_xmp[columns_to_interpolate].astype(float).interpolate().bfill().ffill().astype(str)

    return df_xmp


def write_values(df_towrite, columns, xmp_directory):
    # row-by-row in the df is the same as xmpfile-by-xmpfile
    for row, value in df_towrite.iterrows():
        print('Writing ' + row)
        xmpfullpath = os.path.join(xmp_directory, row) + '.xmp'
        with open(xmpfullpath, 'r') as f:
            xmptext = f.read()

        for column in columns:
            value_to_write = value[column]
            column_list = column.split(' ')
            xmp_text_search = rf'{column_list[0]}:{column_list[1]}="[\d+-.]*"'
            xmp_text_sub = rf'{column_list[0]}:{column_list[1]}="{value_to_write}"'
            if re.search(xmp_text_search, xmptext):
                xmptext = re.sub(xmp_text_search, xmp_text_sub, xmptext)
            else:
                xmp_text_addition = '   \n' + xmp_text_sub
                find_rdf = r'<rdf:Description.*?crs:'
                found_rdf = re.search(find_rdf, xmptext, re.DOTALL)
                add_pos = found_rdf.regs[0][1] - 7
                xmptext = xmptext[:add_pos] + xmp_text_addition + xmptext[add_pos:]
            
        with open (xmpfullpath, 'w') as f:
            f.write(xmptext)
import os

def fsearch(path='.', suffix='', include='', exclude=None, abspath=True, deep=True, hidden=False, short_search=False, folder_only=False, sort_level=-2):
    """search for files with customized constraints

    path: path to starting searching (default: current working directory)
    suffix: search for files end with given suffix (default: all types)
    include: search for files that include certain character or phrase (default: '', not applied)
    exclude: search for files that exclude certain character or phrase (default: None, not applied)
    abspath: set true to return absolute paths of searched files (default: True)
    deep: set true to perform deep search (search in subdirectories, default: True))
    hidden: set true to return hidden files
    sort_level: sort according to certain part from the path, example: "/b/c/d.txt"-->"/-3/-2/-1" (default: -2, search according to file names)
    """
    out = []
    root = os.path.abspath(path)
    abs_paths = [root+'/'+name for name in os.listdir(root)]

    for name in abs_paths:
        if hidden==False:
            if not os.path.basename(name).startswith('.'):
                if os.path.isfile(name):
                    short = name
                    if short_search==True:
                        short = os.path.basename(name)
                    if short.endswith(suffix) and short.find(include)>=0:
                        if exclude!=None:
                            if short.find(exclude)<0:
                                out.append(name)
                        else: out.append(name)
                elif os.path.isdir(name) and deep==True:
                    out = out + fsearch(path=name, suffix=suffix, include=include, exclude=exclude, abspath=True, deep=deep, hidden=hidden, short_search=short_search, folder_only=False, sort_level=sort_level)
        else:
            if os.path.isfile(name):
                short = name
                if short_search==True:
                    short = os.path.basename(name)
                if short.endswith(suffix) and short.find(include)>=0:
                    if exclude!=None:
                        if short.find(exclude)<0:
                            out.append(name)
                    else: out.append(name)
            elif os.path.isdir(name) and deep==True:
                out = out + fsearch(path=name, suffix=suffix, include=include, exclude=exclude, abspath=True, deep=deep, hidden=hidden, short_search=short_search, folder_only=False, sort_level=sort_level)

    out.sort(key=lambda x:x.split('/')[sort_level])
    if folder_only == True:
        out = list(set([os.path.dirname(name) for name in out]))
    if abspath == False:
        out = [os.path.basename(name) for name in out]
    return out



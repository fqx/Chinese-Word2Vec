import glob, codecs, re, gzip
import jieba_fast as jieba

pdpath = '/home/fqx/Documents/pd-corpus/**/*.txt'
corpuspath = 'corpus/pd-aio.txt.gz'
paragraphbreak = re.compile('[‖　]')
linebreak = re.compile('[【】。！？… ]')
jieba.load_userdict('names.txt')
jieba.enable_parallel(4)
aiofile = gzip.open(corpuspath, 'wt', encoding='utf-8')

pdfiles = glob.glob(pdpath,recursive=True)

for addr in pdfiles:
    print('Processing %s' % addr)
    try:
        file = codecs.open(addr, 'r', 'GB18030')
        lines = file.readlines()
        file.close()
    except UnicodeDecodeError:
        print('Decoding Error!')
        continue

    for line in lines:
        paras = re.split(paragraphbreak,line)
        for para in paras:
            reallines = re.split(linebreak,para)
            for realline in reallines:
                if len(realline) > 19:
                    words = jieba.cut(realline)
                    realwords = ' '.join(words)
                    aiofile.write('%s\n' % realwords)
                else:
                    pass

aiofile.close()

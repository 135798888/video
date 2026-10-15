import re,shutil,sys
d=sys.argv[1]
html=open(d+'/index.html').read()
chars=set(c for c in html if ord(c)>0x2e80)|set('，。？！——、：·《》')
def subset(pkg,w,fam):
    css=open(f'{d}/node_modules/@fontsource/{pkg}/{w}.css').read()
    out=[]
    for block in re.findall(r'@font-face\s*{[^}]*}',css):
        ur=re.search(r'unicode-range:\s*([^;]+);',block).group(1)
        rng=[]
        for p in ur.split(','):
            p=p.strip()[2:]; a,b=(p.split('-')+[None])[:2]; rng.append((int(a,16),int(b or a,16)))
        if any(any(a<=ord(c)<=b for a,b in rng) for c in chars):
            f=re.search(r"url\(\./files/([^)]+\.woff2)\)",block).group(1)
            shutil.copy(f'{d}/node_modules/@fontsource/{pkg}/files/{f}',f'{d}/fonts/{f}')
            out.append(re.sub(r", url\(\./files/[^)]+\.woff\) format\('woff'\)","",block).replace('./files/','fonts/'))
    return out
blocks=subset('noto-serif-sc','600','Noto Serif SC')+subset('ma-shan-zheng','400','Ma Shan Zheng')
blocks+=['@font-face { font-family: "Source Serif 4"; font-style: italic; font-weight: 400; src: url(fonts/source-serif-4-latin-400-italic.woff2) format("woff2"); }',
 '@font-face { font-family: "Source Serif 4"; font-style: normal; font-weight: 400; src: url(fonts/source-serif-4-latin-400-normal.woff2) format("woff2"); }',
 '@font-face { font-family: "Source Serif 4"; font-style: normal; font-weight: 600; src: url(fonts/source-serif-4-latin-600-normal.woff2) format("woff2"); }']
start=html.index('/*FONTS*/'); end=html.index('/*FONTS_END*/')+len('/*FONTS_END*/') if '/*FONTS_END*/' in html else start+len('/*FONTS*/')
html=html[:start]+'/*FONTS*/\n      '+'\n      '.join(blocks)+'\n      /*FONTS_END*/'+html[end:]
open(d+'/index.html','w').write(html)
print(len(chars),'chars',len(blocks),'blocks')

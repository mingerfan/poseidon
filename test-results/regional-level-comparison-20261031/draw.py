"""Draw the measured Q34 comparison as editable SVG diagrams."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
COLORS = {
    'green': ('#edf8f2', '#20805b'),
    'blue': ('#edf4ff', '#3266b5'),
    'red': ('#fff0ed', '#bd4b38'),
    'purple': ('#f2effc', '#7556b4'),
    'gray': ('#f2f4f7', '#627084'),
}


def start(height, title, subtitle):
    return [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" viewBox="0 0 1440 {height}">
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#738096"/></marker></defs>
<style>text{{font-family:'Noto Sans CJK SC',sans-serif;fill:#17283e}}.title{{font-size:34px;font-weight:700}}.sub{{font-size:21px;fill:#59677c}}.label{{font-size:24px;font-weight:700}}.state{{font-size:27px;font-weight:700}}.note{{font-size:21px;fill:#45556d}}.small{{font-size:18px;fill:#637189}}</style>
<rect width="1440" height="{height}" fill="#f8fafc"/>
<text x="42" y="56" class="title">{escape(title)}</text>
<text x="42" y="96" class="sub">{escape(subtitle)}</text>''']


def txt(svg, x, y, text, cls='note', anchor='start', color=None):
    style = f' style="fill:{color}"' if color else ''
    svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}"{style}>{escape(text)}</text>')


def box(svg, x, y, title, state, color='blue', width=244):
    fill, stroke = COLORS[color]
    svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="106" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    txt(svg, x+width/2, y+36, title, 'note', 'middle')
    txt(svg, x+width/2, y+78, state, 'state', 'middle', stroke)


def arrow(svg, a, b, y):
    svg.append(f'<path d="M{a+250},{y+53} H{b-9}" fill="none" stroke="#738096" stroke-width="2.5" marker-end="url(#arrow)"/>')


def finish(svg, name):
    svg.append('</svg>')
    (ROOT/name).write_text('\n'.join(svg)+'\n')


svg = start(680, '① 单次ReLU已经追平；差别在Boot之前是否用完剩余Q',
            '公平比较：初始同为Q34/s40 → stem消耗3Q → 首个ReLU消耗22Q → Q9/s40。以下从这里开始。')
txt(svg, 44, 151, '手动阶段约束方案：先做卷积，再Boot', 'label', color=COLORS['green'][1])
y = 174
for x, title, state, color in [
    (44, '首个ReLU出口', 'Q9 / s40', 'green'),
    (321, '卷积：消耗3Q', 'Q6 / s40', 'green'),
    (598, '第1次Boot', 'Q31 / s40', 'purple'),
    (1152, '下一个ReLU：22Q', 'Q9 / s40', 'green'),
]:
    box(svg, x, y, title, state, color)
for a, b in [(44,321),(321,598),(598,1152)]:
    arrow(svg,a,b,y)
txt(svg, 996, 211, '这次ReLU从Q31开始', 'small', 'middle')
txt(svg, 44, 333, '当前自动方案：先Boot，再做卷积', 'label', color=COLORS['red'][1])
y = 356
for x, title, state, color in [
    (44, '首个ReLU出口', 'Q9 / s40', 'blue'),
    (321, 'ModSwitch：丢掉3Q', 'Q6 / s40', 'red'),
    (598, '第1次Boot', 'Q31 / s40', 'purple'),
    (875, '卷积：消耗3Q', 'Q28 / s40', 'blue'),
    (1152, '下一个ReLU：22Q', 'Q6 / s40', 'red'),
]:
    box(svg,x,y,title,state,color)
for a,b in zip([44,321,598,875],[321,598,875,1152]):
    arrow(svg,a,b,y)
svg.append('<rect x="42" y="505" width="1356" height="125" rx="14" fill="white" stroke="#d9e0e9"/>')
txt(svg, 64, 541, '物理计划已确认：第一次Boot前，Q9直接ModSwitch到Q6，确实丢掉3Q。', 'label')
txt(svg, 64, 577, '两边每段“卷积3Q + ReLU22Q”的算术开销相同；Boot位置不同，让ReLU出口一直相差3Q。')
txt(svg, 64, 610, '这是有效Q预算的损失，不是额外3次RS。图中只画主路径；残差分支见文字解释。', 'small')
txt(svg, 44, 657, '依据：regional-q34/scale-trace.json；Q34物理计划Boot入口审计；历史manual34的阶段约束与18个Boot位置。', 'small')
finish(svg,'boot-position.svg')

svg = start(880, '② 为什么自动18次Boot时输出s160，而手动18次能输出s40？',
            '都看第18次Boot附近。手动的末次卷积已在Boot之前完成；自动还要在Boot之后做。')
txt(svg, 44, 148, '手动：最后一个ReLU从Q31开始，尾部拿到Q9', 'label', color=COLORS['green'][1])
y = 170
for x,title,state,color in [
    (44,'第18次Boot出口','Q31 / s40','purple'),
    (413,'末次ReLU：22Q','Q9 / s40','green'),
    (782,'池化/分类尾部：4Q','Q5 / s40','green'),
    (1152,'直接输出','18次Boot / s40','green'),
]:
    box(svg,x,y,title,state,color)
for a,b in [(44,413),(413,782),(782,1152)]:
    arrow(svg,a,b,y)
txt(svg, 44, 328, '自动，允许高scale输出：尾部保留乘积scale，省掉末端RS', 'label', color=COLORS['blue'][1])
y = 350
for x,title,state,color in [
    (44,'第18次Boot出口','Q31 / s40','purple'),
    (321,'末次卷积：3Q','Q28 / s40','blue'),
    (598,'末次ReLU：22Q','Q6 / s40','blue'),
    (875,'尾部三层CP，无RS','Q6 / s160','blue'),
    (1152,'直接输出','18次Boot / s160','blue'),
]:
    box(svg,x,y,title,state,color)
for a,b in zip([44,321,598,875],[321,598,875,1152]):
    arrow(svg,a,b,y)
txt(svg, 44, 509, '自动，要求最终s40：Q6不足以完成尾部归一化，于是多一次Boot', 'label', color=COLORS['red'][1])
y = 531
for x,title,state,color in [
    (44,'末次ReLU出口','Q6 / s40','red'),
    (413,'补第19次Boot','Q31 / s40','purple'),
    (782,'同一尾部：5Q','Q26 / s40','red'),
    (1152,'最终输出','19次Boot / s40','red'),
]:
    box(svg,x,y,title,state,color)
for a,b in [(44,413),(413,782),(782,1152)]:
    arrow(svg,a,b,y)
svg.append('<rect x="42" y="681" width="1356" height="144" rx="14" fill="white" stroke="#d9e0e9"/>')
txt(svg, 64, 718, 's160的来源：40 + 40 + 40 + 40 = 160；三个明文编码scale均至少40。', 'label')
txt(svg, 64, 753, '要回到s40，至少要消掉120位；Q6→Q5只消掉31.99位，因此放宽到Q5仍不够。')
txt(svg, 64, 788, '真实链也影响尾部成本：Q9移除4个模数共127.96位；Q31移除4个仅119.985位，需要第5个。', 'small')
txt(svg, 44, 855, '“18次/s160”是放开最终scale后的自然结果；图示为计划与状态校验结果，不是整网GPU延迟或密态精度测量。', 'small')
finish(svg,'terminal-scale.svg')

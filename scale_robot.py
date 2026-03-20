import xml.etree.ElementTree as ET

FACTOR = 1.0 / 1.5

def scale_vector(vec_str, scale):
    vals = [float(x) * scale for x in vec_str.strip().split()]
    return ' '.join([f"{v:.5g}" for v in vals])

tree = ET.parse('/home/acer/Clawdrobot/pal_tiago_dual/tiago_dual.xml')
root = tree.getroot()

for elem in root.iter():
    if 'pos' in elem.attrib:
        elem.attrib['pos'] = scale_vector(elem.attrib['pos'], FACTOR)
    if 'size' in elem.attrib:
        elem.attrib['size'] = scale_vector(elem.attrib['size'], FACTOR)
        
    if elem.tag == 'mesh':
        if 'scale' in elem.attrib:
            elem.attrib['scale'] = scale_vector(elem.attrib['scale'], FACTOR)
        else:
            elem.attrib['scale'] = f"{FACTOR:.5g} {FACTOR:.5g} {FACTOR:.5g}"
            
    if elem.tag == 'joint' and elem.attrib.get('type') == 'slide':
        if 'range' in elem.attrib:
            elem.attrib['range'] = scale_vector(elem.attrib['range'], FACTOR)
            
    if elem.tag == 'inertial':
        if 'mass' in elem.attrib:
            elem.attrib['mass'] = str(round(float(elem.attrib['mass']) * (FACTOR**3), 5))
        if 'diaginertia' in elem.attrib:
            elem.attrib['diaginertia'] = scale_vector(elem.attrib['diaginertia'], FACTOR**5)

tree.write('tiago_dual_scaled.xml', encoding='utf-8', xml_declaration=True)
print(f"Thành công! Đã tạo file 'tiago_dual_scaled.xml' nhỏ hơn 1.5 lần.")
import xml.etree.ElementTree as ET

# Hệ số thu nhỏ (Giảm 1.5 lần = nhân với 1/1.5)
FACTOR = 1.0 / 1.5

def scale_vector(vec_str, scale):
    """Hàm nhân tất cả các số trong một chuỗi vector với hệ số scale"""
    vals = [float(x) * scale for x in vec_str.strip().split()]
    return ' '.join([f"{v:.5g}" for v in vals])

# 1. Đọc file robot gốc của bạn
tree = ET.parse('/home/acer/Clawdrobot/pal_tiago_dual/tiago_dual.xml')
root = tree.getroot()

# 2. Quét qua toàn bộ thẻ XML để thu nhỏ
for elem in root.iter():
    # Thu nhỏ khoảng cách (pos) và kích thước hình học (size)
    if 'pos' in elem.attrib:
        elem.attrib['pos'] = scale_vector(elem.attrib['pos'], FACTOR)
    if 'size' in elem.attrib:
        elem.attrib['size'] = scale_vector(elem.attrib['size'], FACTOR)
        
    # Thu nhỏ file 3D (mesh)
    if elem.tag == 'mesh':
        if 'scale' in elem.attrib:
            elem.attrib['scale'] = scale_vector(elem.attrib['scale'], FACTOR)
        else:
            elem.attrib['scale'] = f"{FACTOR:.5g} {FACTOR:.5g} {FACTOR:.5g}"
            
    # Thu nhỏ giới hạn chuyển động của các khớp trượt (slide)
    if elem.tag == 'joint' and elem.attrib.get('type') == 'slide':
        if 'range' in elem.attrib:
            elem.attrib['range'] = scale_vector(elem.attrib['range'], FACTOR)
            
    # Thu nhỏ khối lượng (tỷ lệ mũ 3) và quán tính (tỷ lệ mũ 5) để robot không bị nổ
    if elem.tag == 'inertial':
        if 'mass' in elem.attrib:
            elem.attrib['mass'] = str(round(float(elem.attrib['mass']) * (FACTOR**3), 5))
        if 'diaginertia' in elem.attrib:
            elem.attrib['diaginertia'] = scale_vector(elem.attrib['diaginertia'], FACTOR**5)

# 3. Lưu ra file robot mới
tree.write('tiago_dual_scaled.xml', encoding='utf-8', xml_declaration=True)
print(f"Thành công! Đã tạo file 'tiago_dual_scaled.xml' nhỏ hơn 1.5 lần.")
# -*- encoding: utf8 -*-
import csv
import json
from io import open


def clean_data(filename, output_file):
    with open(filename, "r", encoding='utf-8') as f, open(output_file, "w", encoding='utf-8') as f2:
        data = f.read()
        print type(data)
        j = json.loads(data)
        total_comments = len(j)
        count = 0
        col1 = []; col2 = []
        for item in j:
            try:
                id1 = item['id']
                id1 = int(id1)
                message = item['message']
                col1.append(message)
                label= item['labels'][0].values()
                sentiment = label[0][0]['name']
                name1 = label[1][0]['name']
                try:
                    name2 = label[1][1]['name']
                except:
                    pass
                o = u"other"; t0 = u"tiêu cực"; t1 = u"trung lập"; t2 = u"tích cực"
                clsp = u"chất lượng sản phẩm"; o2 = u"other"; nd = u"nội dung môn học"
                dv = u"dịch vụ"; gv = u"giảng viên: giảng viên chuyên môn hay GVHD"
                ctsv = u"hoạt động của công tác sinh viên"; kt = u"hệ thống kỹ thuật: đường truyền, UX"
                cssv = u"phản hồi, tư vấn, chăm sóc sinh viên"
                if name1.lower() == o and sentiment.lower() == t2:
                    f2.write("OT2" + "|" + message +"\n")
                if name1.lower() == o and sentiment.lower() == t1:
                    f2.write("OT1" + "|" + message + "\n")
                if name1.lower() == o and sentiment.lower() == t0:
                    f2.write("OT0" + "|" + message + "\n")

                if name1.lower() == clsp and name2 == o2 and sentiment.lower() == t2:
                    f2.write("CLOT2" + " | " + message + "\n")
                if name1.lower() == clsp and name2 == o2 and sentiment.lower() == t1:
                    f2.write("CLOT1" + " | " + message + "\n")
                if name1.lower() == clsp and name2 == nd and sentiment.lower() == t0:
                    f2.write("CLNDT0" + " | " + message + "\n")
                if name1.lower() == clsp and name2 == nd and sentiment.lower() == t1:
                    f2.write("CLNDT1" + " | " + message + "\n")
                if name1.lower() == clsp and name2 == nd and sentiment.lower() == t2:
                    f2.write("CLNDT2" + " | " + message + "\n")

                if name1.lower() == dv and name2 == o2 and sentiment.lower() == t2:
                    f2.write("DVOT2" + " | " + message + "\n")
                if name1.lower() == dv and name2 == o2 and sentiment.lower() == t1:
                    f2.write("DVOT1" + " | " + message + "\n")
                if name1.lower() == dv and name2 == o2 and sentiment.lower() == t0:
                    f2.write("DVOT0" + " | " + message + "\n")

                if name1.lower() == dv and name2 == gv and sentiment.lower() == t2:
                    f2.write("DVGV2" + " | " + message + "\n")
                if name1.lower() == dv and name2 == ctsv and sentiment.lower() == t2:
                    f2.write("DVCTSV2" + " | " + message + "\n")

                if name1.lower() == dv and name2 == kt and sentiment.lower() == t0:
                    f2.write("DVKT0" + " | " + message + "\n")
                if name1.lower() == dv and name2 == kt and sentiment.lower() == t2:
                    f2.write("DVKT2" + " | " + message + "\n")
                if name1.lower() == dv and name2 == cssv and sentiment.lower() == t0:
                    f2.write("DVCSSV0" + " | " + message + "\n")
                if name1.lower() == dv and name2 == cssv and sentiment.lower() == t2:
                    f2.write("DVCSSV2" + " | " + message + "\n")
                # f2.write(name1 + " " + name2 + " " + sentiment + " | "+ message)

            except Exception as e:
                # print e.message
                count += 1
    print count

def clean_data2(filename, output_file):
    with open(filename, "r", encoding='utf-8') as f, open(output_file, "w", encoding='utf-8') as f2:
        data = f.read()
        j = json.loads(data)
        er = 0
        for item in j:
            try:
                what = item['what']
                message = what['message']
                # print message
                labels = item['labels']
                l_list = labels[0]['list'][0]  # lay tu dau { sau "list: ["
                l = l_list.values() #[[{u'id': 0, u'name': u'Trung l\u1eadp'}], [{u'id': 0, u'name': u'ch\u1ea5t l\u01b0\u1ee3ng s\u1ea3n ph\u1ea9m'}, {u'id': 0, u'name': u'n\u1ed9i dung m\xf4n h\u1ecdc'}]]
                sentiment = l[0][0]['name']
                name1 = l[1][0]['name']
                try:
                    name2 = l[1][1]['name']
                except:
                    pass
                o = u"other";
                t0 = u"tiêu cực";
                t1 = u"trung lập";
                t2 = u"tích cực"
                clsp = u"chất lượng sản phẩm";
                o2 = u"other";
                nd = u"nội dung môn học"
                dv = u"dịch vụ";
                gv = u"giảng viên: giảng viên chuyên môn hay GVHD"
                ctsv = u"hoạt động của công tác sinh viên";
                kt = u"hệ thống kỹ thuật: đường truyền, UX"
                cssv = u"phản hồi, tư vấn, chăm sóc sinh viên"
                if name1.lower() == o and sentiment.lower() == t2:
                    f2.write("OT2" + "|" + message + "\n")
                if name1.lower() == o and sentiment.lower() == t1:
                    f2.write("OT1" + "|" + message + "\n")
                if name1.lower() == o and sentiment.lower() == t0:
                    f2.write("OT0" + "|" + message + "\n")

                if name1.lower() == clsp and name2 == o2 and sentiment.lower() == t2:
                    f2.write("CLOT2" + " | " + message + "\n")
                if name1.lower() == clsp and name2 == o2 and sentiment.lower() == t1:
                    f2.write("CLOT1" + " | " + message + "\n")

                if name1.lower() == clsp and name2 == nd and sentiment.lower() == t0:
                    f2.write("CLNDT0" + " | " + message + "\n")
                if name1.lower() == clsp and name2 == nd and sentiment.lower() == t1:
                    f2.write("CLNDT1" + " | " + message + "\n")
                if name1.lower() == clsp and name2 == nd and sentiment.lower() == t2:
                    f2.write("CLNDT1" + " | " + message + "\n")

                if name1.lower() == dv and name2 == o2 and sentiment.lower() == t2:
                    f2.write("DVOT2" + " | " + message + "\n")
                if name1.lower() == dv and name2 == o2 and sentiment.lower() == t1:
                    f2.write("DVOT1" + " | " + message + "\n")
                if name1.lower() == dv and name2 == o2 and sentiment.lower() == t0:
                    f2.write("DVOT0" + " | " + message + "\n")

                if name1.lower() == dv and name2 == gv and sentiment.lower() == t2:
                    f2.write("DVGV2" + " | " + message + "\n")
                if name1.lower() == dv and name2 == ctsv and sentiment.lower() == t2:
                    f2.write("DVCTSV2" + " | " + message + "\n")

                if name1.lower() == dv and name2 == kt and sentiment.lower() == t0:
                    f2.write("DVKT0" + " | " + message + "\n")
                if name1.lower() == dv and name2 == kt and sentiment.lower() == t2:
                    f2.write("DVKT2" + " | " + message + "\n")
                if name1.lower() == dv and name2 == cssv and sentiment.lower() == t0:
                    f2.write("DVCSSV0" + " | " + message + "\n")
                if name1.lower() == dv and name2 == cssv and sentiment.lower() == t2:
                    f2.write("DVCSSV2" + " | " + message + "\n")

            except:
                er += 1
        print er



if __name__ == '__main__':
    # clean_data('data/labels_uni_forum_report_tnu.json','general_data/cleaned_report.txt')
    clean_data2('data/labeled_expressions_forum_report_hou2.json', 'general_data/cleaned_report2.txt' )

#


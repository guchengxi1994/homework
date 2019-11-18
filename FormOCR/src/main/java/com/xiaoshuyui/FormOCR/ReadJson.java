package com.xiaoshuyui.FormOCR;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import net.sf.json.JSONArray;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ReadJson {

    public static String readFile(String filepath) {
        String jsonStr = "";

        try {
//            File jsonFile = new File("D:\\formOCR\\jsonData.txt");
            File jsonFile = new File(filepath);
            FileReader fileReader = new FileReader(jsonFile);
            Reader reader = new InputStreamReader(new FileInputStream(jsonFile), "utf-8");

            int ch = 0;
            StringBuffer sb = new StringBuffer();
            while ((ch = reader.read()) != -1) {
                sb.append((char) ch);
            }
            fileReader.close();
            reader.close();
            jsonStr = sb.toString();
            return jsonStr;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

    }

    public static void main(String[] args) {
        String filePath = "D:\\formOCR\\jsonData.txt";
        String s = readFile(filePath);
//        System.out.println(s);

        JSONObject jsonObject = JSON.parseObject(s);
//        JSONObject forms = (JSONObject) JSONObject.parse(jsonObject.get("forms").toString());
//        System.out.println(jsonObject.get("forms").toString());
        String forms = jsonObject.get("forms").toString();
        forms = forms.substring(1, forms.length() - 1);
//        System.out.println(forms);
        JSONObject temp = JSON.parseObject(forms);
        String body = temp.get("body").toString();
//        System.out.println(body);

        JSONArray jsonArray = JSONArray.fromObject(body);
//        System.out.println(jsonArray.get(0).toString());

        List<Param> list = new ArrayList<Param>();
        for (Object o : jsonArray) {

//            System.out.println(jsonArray.get(i).toString());
            JSONObject temp0 = JSON.parseObject(o.toString());
            Param p = new Param(temp0.getString("column"), temp0.getString("row"), temp0.getString("word"), temp0.getString("rect"));
            list.add(p);
        }

//        for (Object o : list
//        ) {
//            System.out.println(o);
//        }

        Collections.sort(list, new Comparator<Param>() {
            public int compare(Param o1, Param o2) {
//                return 0;

                String c1 = o1.getColumn();
                c1 = c1.substring(1, c1.length() - 1);
                String c2 = o2.getColumn();
                c2 = c2.substring(1, c2.length() - 1);
                String r1 = o1.getRow();
                r1 = r1.substring(1, r1.length() - 1);
                String r2 = o2.getRow();
                r2 = r2.substring(1, r2.length() - 1);

                if (c1.length() > 1 || c2.length() > 1 || r1.length() > 1 || r2.length() > 1) {

                    return 0;
                } else {
                    int ir1 = Integer.parseInt(r1);
                    int ir2 = Integer.parseInt(r2);
                    int ic1 = Integer.parseInt(c1);
                    int ic2 = Integer.parseInt(c2);

                    if (ir1 == ir2) {
                        if (ic1 - ic2 > 0) {
                            return 1;
                        } else {
                            return -1;
                        }
//                        return ic1 - ic2;
                    } else {
                        if (ir1 - ir2 > 0) {
                            return 1;
                        } else {
                            return -1;
                        }
//                        return ir1 - ir2;
                    }

                }
            }
        });

//        System.out.println(list.toString());

        for (Object o : list
        ) {
            System.out.println(o);

        }

    }
}

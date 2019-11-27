package com.xiaoshuyui.kaidian.controller;


import com.xiaoshuyui.kaidian.entity.HelloEntity;
import com.xiaoshuyui.kaidian.utils.StatusCode;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.sql.SQLException;

@Controller
@ResponseBody
@RequestMapping("/test")
public class HelloController {

    @Autowired
    private Environment env;

    @RequestMapping("/hello")

    public HelloEntity Hello() {
        HelloEntity helloEntity = new HelloEntity();
        try {
//            StatusCode statusCode = StatusCode.SUCCESS;
            helloEntity.setMessage("hello,开店的");
            helloEntity.setCode(StatusCode.SUCCESS);

//            helloEntity.setCode(Integer.parseInt(env.getProperty("${application.SUCCESS.code}")));
            helloEntity.setData(null);
//            throw new SQLException();
        } catch (Exception e) {
            e.printStackTrace();
            helloEntity.setMessage("error");
            helloEntity.setCode(StatusCode.ERROR);
            helloEntity.setData(null);
        }

        return helloEntity;

    }
//    public String Hello(){
//        return "hello 开店";
//    }
}


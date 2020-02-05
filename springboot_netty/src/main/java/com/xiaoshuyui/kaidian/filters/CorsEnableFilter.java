package com.xiaoshuyui.kaidian.filters;


import org.apache.commons.lang.StringUtils;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;


import javax.servlet.*;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class CorsEnableFilter implements Filter {

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {

    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        HttpServletResponse httpServletResponse = (HttpServletResponse) response;
        HttpServletRequest httpServletRequest = (HttpServletRequest) request;
        String domain = httpServletRequest.getHeader("Origin");
        String method = httpServletRequest.getMethod();
        httpServletResponse.setHeader("Access-Control-Allow-Origin", domain);
        httpServletResponse.setHeader("Access-Control-Allow-Methods", method);
        httpServletResponse.setHeader("Access-Control-Allow-Credentials", "true");
        httpServletResponse.setHeader("Access-Control-Allow-Headers",
                "Client-Info, Captcha, X-Requested-With, Authorization, Content-Type, Credential, X-XSRF-TOKEN");

        if (StringUtils.equalsIgnoreCase(httpServletRequest.getMethod(), "OPTIONS")) {
            httpServletResponse.setStatus(HttpServletResponse.SC_OK);
        } else {
            chain.doFilter(request, response);
        }
    }

    @Override
    public void destroy() {

    }
}

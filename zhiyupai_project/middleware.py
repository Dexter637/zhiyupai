import re
from django.conf import settings
from django.http import HttpResponseForbidden

class SecurityMiddleware:
    """
    自定义安全中间件，用于增强API安全性
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        # 编译白名单URL模式
        self.public_urls = [re.compile(pattern) for pattern in [
            r'^/api/auth/',  # 认证相关的URL
            r'^/api/users/register/$',  # 注册URL
            r'^/api/users/login/$',  # 登录URL
            r'^/admin/',  # 管理后台URL
            r'^/api-auth/',  # DRF认证URL
            r'^/api/wearable/upload/$',  # 可穿戴设备数据上传URL
        ]]
    
    def __call__(self, request):
        # 检查是否为API请求
        if request.path.startswith('/api/'):
            # 检查是否为公开URL
            is_public_url = any(pattern.match(request.path) for pattern in self.public_urls)
            
            # 如果不是公开URL，检查认证
            if not is_public_url and not request.user.is_authenticated:
                return HttpResponseForbidden("认证失败，请先登录")
            
            # 检查API请求速率限制（可以根据需要实现）
            # 这里可以添加IP限制、用户请求频率限制等
            
            # 检查请求来源（可选，如果需要更严格的来源控制）
            if settings.DEBUG is False and request.META.get('HTTP_REFERER'):
                allowed_referers = getattr(settings, 'ALLOWED_REFERERS', [])
                if allowed_referers and not any(request.META.get('HTTP_REFERER').startswith(ref) for ref in allowed_referers):
                    return HttpResponseForbidden("请求来源不被允许")
        
        # 继续处理请求
        response = self.get_response(request)
        
        # 添加安全响应头
        if not settings.DEBUG:
            response['X-Content-Type-Options'] = 'nosniff'
            response['X-Frame-Options'] = 'DENY'
            response['X-XSS-Protection'] = '1; mode=block'
            response['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            response['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' 'unsafe-eval';"
        
        return response

class APILoggingMiddleware:
    """
    API请求日志中间件，记录API请求和响应
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # 只记录API请求
        if request.path.startswith('/api/'):
            # 记录请求信息
            request_log = {
                'path': request.path,
                'method': request.method,
                'user_id': request.user.id if request.user.is_authenticated else None,
                'ip': self.get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                'query_params': dict(request.GET.items()),
            }
            
            # 可以将请求日志保存到数据库或日志文件
            # 这里简单打印到控制台
            if settings.DEBUG:
                print(f"API请求: {request_log}")
        
        # 处理请求
        response = self.get_response(request)
        
        # 记录响应信息（可选）
        if request.path.startswith('/api/') and settings.DEBUG:
            response_log = {
                'path': request.path,
                'status_code': response.status_code,
                'content_length': len(response.content) if hasattr(response, 'content') else 0,
            }
            print(f"API响应: {response_log}")
        
        return response
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from models import UserProfile

class PhoneBackend(ModelBackend):
    """自定义认证后端，支持手机号登录"""
    
    def authenticate(self, request, username=None, password=None, **kwargs):
        User = get_user_model()
        
        # 首先尝试使用默认的用户名认证
        if username is not None:
            try:
                user = User.objects.get(username=username)
                if user.check_password(password):
                    return user
            except User.DoesNotExist:
                pass
        
        # 如果用户名认证失败，尝试手机号认证
        phone = kwargs.get('phone') or username
        if phone:
            try:
                # 通过手机号查找用户档案
                profile = UserProfile.find({"phone": phone})
                if profile:
                    # 通过用户ID查找Django用户
                    user = User.objects.get(id=profile[0]['user_id'])
                    if user.check_password(password):
                        return user
            except (User.DoesNotExist, ValueError, IndexError):
                pass
        
        return None
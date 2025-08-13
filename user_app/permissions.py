from rest_framework import permissions

class IsOwnerOrAdmin(permissions.BasePermission):
    """
    自定义权限类，只允许对象的所有者或管理员访问
    """
    
    def has_permission(self, request, view):
        # 验证用户是否已认证
        return request.user and request.user.is_authenticated
    
    def has_object_permission(self, request, view, obj):
        # 允许管理员访问任何对象
        if request.user.is_staff:
            return True
            
        # 检查对象是否有user_id属性
        if hasattr(obj, 'user_id'):
            return str(obj.user_id) == str(request.user.id)
        
        # 检查对象是否为字典类型且包含user_id键
        if isinstance(obj, dict) and 'user_id' in obj:
            return str(obj['user_id']) == str(request.user.id)
        
        # 检查对象是否为Django用户模型
        if hasattr(obj, 'id'):
            return obj.id == request.user.id
            
        # 默认拒绝访问
        return False

class IsAdminUser(permissions.BasePermission):
    """
    自定义权限类，只允许管理员访问
    """
    
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated and request.user.is_staff
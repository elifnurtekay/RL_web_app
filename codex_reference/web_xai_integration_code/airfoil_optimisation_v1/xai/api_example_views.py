from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .xai_service import XAIService
@api_view(['POST'])
def explain_existing_optimization_result(request):
    algorithm=request.data.get('algorithm'); user_input=request.data.get('user_input',{}); trajectory=request.data.get('trajectory',[]); optimized_result=request.data.get('optimized_result')
    if not algorithm: return Response({'error':'algorithm gerekli: ppo/td3/sac'}, status=400)
    if not trajectory: return Response({'error':'trajectory gerekli'}, status=400)
    # TODO: mevcut surrogate solver fonksiyonunuza bağlayın: solver_fn(cst, aoa, re) -> dict
    solver_fn=None
    service=XAIService(artifact_root=settings.BASE_DIR/'xai_artifacts')
    return Response(service.explain_optimized_airfoil(algorithm, user_input, trajectory, optimized_result, solver_fn))

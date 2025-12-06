import os
import argparse

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos, incluyendo
    la selección de textura inicial.
    """
    parser = argparse.ArgumentParser(
        description='Superposición de prenda con detección de pose',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--debug', '-d', action='store_true', 
                       help='Activar modo debug')
    
    parser.add_argument('--use-3d', action='store_true', 
                       help='Usar modelo 3D')
    
    parser.add_argument('--simplify-factor', type=int, default=3,
                       help='Factor de simplificación 3D (mayor = mejor rendimiento)')
    
    parser.add_argument('--render-mode',
            type=str,
            default='wireframe',
            choices=['textured', 'wireframe'],
            help='Modo de renderizado 3D'
        )
    
    # Texturas disponibles dinámicamente (se cargarán de archivos)
    parser.add_argument('--texture',
            type=str,
            default=None,  # Ninguna por defecto
            help="""Nombre de la textura a usar inicialmente.
Debe corresponder a un archivo en el directorio de texturas.
Ejemplos: shirt_black, shirt_white, shirt_blue, shirt_red"""
        )
    
    parser.add_argument('--textures-dir',
            type=str,
            default=os.path.expanduser('~/AR_python/Aumented_Reality/models/textures/'),
            help='Directorio donde buscar texturas'
        )
    
    parser.add_argument('--list-textures',
            action='store_true',
            help='Listar texturas disponibles y salir'
        )
    
    return parser.parse_args()


args = parse_arguments()

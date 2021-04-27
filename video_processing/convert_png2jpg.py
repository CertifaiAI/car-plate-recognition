from PIL import Image
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('image', None, 'path to input image')

def main(argv):
    im = Image.open(FLAGS.image)
    rgb_im = im.convert('RGB')
    rgb_im.save('test.jpeg')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
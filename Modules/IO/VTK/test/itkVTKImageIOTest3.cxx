/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkVTKImageIO.h"
#include "itkTestingMacros.h"


// Specific ImageIO test

int
itkVTKImageIOTest3(int argc, char * argv[])
{

  if (argc != 2)
  {

    std::cerr << "Missing parameters." << std::endl;
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv);
    std::cerr << "  polyDataFile" << std::endl;
    return EXIT_FAILURE;
  }


  const itk::VTKImageIO::Pointer vtkImageIO = itk::VTKImageIO::New();
  // Ensure that the ImageIO does not claim to read a .vtk file with PolyData
  ITK_TEST_EXPECT_EQUAL(vtkImageIO->CanReadFile(argv[1]), false);

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}
